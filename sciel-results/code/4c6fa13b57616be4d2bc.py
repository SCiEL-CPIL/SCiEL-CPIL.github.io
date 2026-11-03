"""Spectral Graph Partitioning with 2-Opt Local Search for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """Compute cost change when swapping positions i and j."""
    x = p[i]
    y = p[j]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    val = np.dot(diff_a, diff_b[p])
    return val - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]

def _spectral_bisection(weights, k=2):
    """
    Spectral bisection of a graph defined by weights matrix.
    Returns a binary assignment vector (0 or 1) for each node.
    """
    n = len(weights)
    if n <= 1:
        return np.zeros(n, dtype=int)
    
    # Ensure symmetry
    W = (weights + weights.T) / 2.0
    
    # Compute degree
    deg = W.sum(axis=1)
    
    # Laplacian L = D - W
    # We want to solve L x = lambda x for the smallest non-zero eigenvalue (Fiedler vector)
    # Using scipy.sparse.linalg.eigsh for efficiency
    
    D = csr_matrix(np.diag(deg))
    L = D - csr_matrix(W)
    
    # Find the smallest non-zero eigenvalue.
    # eigsh with 'SM' (smallest magnitude) is slow. 'SA' (smallest algebraic) is better for symmetric.
    # The smallest eigenvalue is always 0 (constant vector). We want the next one.
    # k=2 to get the two smallest.
    
    try:
        # sigma=0 shift-invert for better convergence on smallest
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
    except Exception:
        # Fallback if sigma=0 fails
        try:
            vals, vecs = eigsh(L, k=2, which='SA')
        except Exception:
            # Last resort: return random split
            return np.random.randint(0, 2, n)
    
    # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
    # eigsh returns in ascending order of eigenvalues (for 'SA' or 'LM' with sigma=0, order might vary, 
    # but typically smallest first).
    # Let's sort to be sure.
    idx = np.argsort(vals)
    fiedler_vec = vecs[:, idx[1]]
    
    # Normalize to mean 0
    fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
    
    # Assign 0 if < 0, 1 if >= 0
    assignment = (fiedler_vec >= 0).astype(int)
    
    # Check balance. If very unbalanced, we might want to adjust, but for QAP 
    # balanced partitions are generally good.
    # If one side is empty (shouldn't happen with Fiedler), handle it.
    if assignment.sum() == 0 or assignment.sum() == n:
        # Random split
        assignment = np.random.randint(0, 2, n)
        
    return assignment

def _spectral_partition_init(a, b, seed=42):
    """
    Construct an initial permutation using recursive spectral bisection.
    """
    n = len(a)
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # We will build a tree of partitions.
    # At each node, we have a set of facilities and a set of locations.
    # We bisect both sets using spectral bisection on their respective matrices.
    # Then we match the two parts.
    
    # To avoid recursion depth issues and overhead, we do a level-order or simple recursive approach.
    # Given n=256, depth is ~8.
    
    # We need to maintain the mapping from original indices to the current sub-problem indices.
    # This is complex to manage directly. 
    # Alternative: 
    # 1. Bisect the full facility graph -> F1, F2
    # 2. Bisect the full location graph -> L1, L2
    # 3. Recursively assign F1 to L1 and F2 to L2? 
    #    Wait, the cost is sum a[i,j] b[p[i], p[j]].
    #    If we split facilities into F1, F2 and locations into L1, L2, 
    #    the optimal assignment likely maps F1 to L1 and F2 to L2 (or vice versa) 
    #    if the intra-cluster flows/distances dominate.
    #    However, we don't know which part of the location graph corresponds to which part of the facility graph.
    #    We can try both mappings or use a heuristic.
    
    # Let's implement a recursive function that takes:
    # - list of facility indices
    # - list of location indices
    # - returns a dict mapping facility_idx -> location_idx
    
    assignment = {}
    
    def recurse(fac_indices, loc_indices):
        if len(fac_indices) == 1:
            assignment[fac_indices[0]] = loc_indices[0]
            return
        
        if len(fac_indices) <= 2:
            # Assign directly
            assignment[fac_indices[0]] = loc_indices[0]
            assignment[fac_indices[1]] = loc_indices[1]
            return
        
        # Extract sub-matrices
        f_idx = np.array(fac_indices)
        l_idx = np.array(loc_indices)
        
        A_sub = a[np.ix_(f_idx, f_idx)]
        B_sub = b[np.ix_(l_idx, l_idx)]
        
        # Spectral bisection on Facilities
        fac_split = _spectral_bisection(A_sub)
        # Spectral bisection on Locations
        loc_split = _spectral_bisection(B_sub)
        
        # Identify the two groups
        fac_group_0 = [f_idx[i] for i in range(len(f_idx)) if fac_split[i] == 0]
        fac_group_1 = [f_idx[i] for i in range(len(f_idx)) if fac_split[i] == 1]
        
        loc_group_0 = [l_idx[i] for i in range(len(l_idx)) if loc_split[i] == 0]
        loc_group_1 = [l_idx[i] for i in range(len(l_idx)) if loc_split[i] == 1]
        
        # Ensure groups are not empty
        if not fac_group_0 or not fac_group_1:
            # Fallback: random split
            mid = len(fac_indices) // 2
            fac_group_0 = fac_indices[:mid]
            fac_group_1 = fac_indices[mid:]
            
        if not loc_group_0 or not loc_group_1:
            mid = len(loc_indices) // 2
            loc_group_0 = loc_indices[:mid]
            loc_group_1 = loc_indices[mid:]
            
        # Decide mapping: Fac Group 0 -> Loc Group 0 or Loc Group 1?
        # Heuristic: Match the "heavier" or "more central" parts?
        # Or simply try one mapping. Since we will do local search, the exact mapping
        # at the top level might be corrected if it's wrong, but it's better to guess right.
        # Let's compute the average row sum of A_sub for each group and B_sub for each group.
        # High flow facilities should go to low distance (central) locations?
        # Actually, in QAP, we want high flow to be close.
        # Let's look at the "centrality" or average distance.
        
        # Simple heuristic: Just map 0->0 and 1->1. 
        # If this is a bad guess, the local search will have to fix it, which might be hard.
        # A better heuristic: 
        # Calculate the total flow within Fac Group 0 and Fac Group 1.
        # Calculate the total distance within Loc Group 0 and Loc Group 1.
        # We want to minimize sum(flow * dist).
        # If Fac Group 0 has high internal flow, it should go to Loc Group with low internal distance.
        
        flow_0 = A_sub[np.ix_(f_idx[fac_split==0], f_idx[fac_split==0])].sum()
        flow_1 = A_sub[np.ix_(f_idx[fac_split==1], f_idx[fac_split==1])].sum()
        
        dist_0 = B_sub[np.ix_(l_idx[loc_split==0], l_idx[loc_split==0])].sum()
        dist_1 = B_sub[np.ix_(l_idx[loc_split==1], l_idx[loc_split==1])].sum()
        
        # If flow_0 > flow_1, we want to put Fac Group 0 in the location group with smaller distance sum.
        # If dist_0 < dist_1, map Fac 0 -> Loc 0.
        # Else map Fac 0 -> Loc 1.
        
        if flow_0 > flow_1:
            if dist_0 < dist_1:
                recurse(fac_group_0, loc_group_0)
                recurse(fac_group_1, loc_group_1)
            else:
                recurse(fac_group_0, loc_group_1)
                recurse(fac_group_1, loc_group_0)
        else:
            if dist_0 < dist_1:
                recurse(fac_group_1, loc_group_0)
                recurse(fac_group_0, loc_group_1)
            else:
                recurse(fac_group_1, loc_group_1)
                recurse(fac_group_0, loc_group_0)
        
    all_fac = list(range(n))
    all_loc = list(range(n))
    recurse(all_fac, all_loc)
    
    # Convert dict to permutation array
    p = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p[i] = assignment[i]
        
    return p

def _local_search_2opt(a, b, p, time_limit, start_time, seed=42):
    """
    2-Opt style local search: iteratively swap pairs (i, j) if it improves cost.
    Uses random sampling of pairs to avoid O(n^2) full scan cost if too slow, 
    but for n=256, O(n^2) is ~32k checks. Each check is O(n) dot product.
    32k * 256 = 8M ops. NumPy can do this in ~10-50ms.
    So a full scan per iteration is feasible.
    """
    n = len(a)
    a_sym = a + a.T
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    rng = random.Random(seed)
    
    while time.time() - start_time < time_limit - 5:
        improved = False
        
        # Full scan of all pairs?
        # n=256 -> 32640 pairs.
        # Let's do a full scan.
        
        # Precompute something?
        # delta = dot(diff_a, diff_b[p]) - ...
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Vectorized full scan?
        # It's tricky to vectorize the swap delta for all pairs at once efficiently without large memory.
        # Loop is safer.
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit - 5:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a_sym, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
            
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Strategy:
    # 1. Generate multiple spectral initializations with different seeds/perturbations.
    # 2. Run 2-opt local search on each.
    # 3. Keep the best.
    
    # How many? 
    # Spectral init is O(n^3) or so due to eigsh. 
    # For n=256, eigsh on 256x256 is fast (~10-50ms).
    # Local search is the bottleneck.
    # Let's try 3-5 islands.
    
    num_islands = 4
    
    for k in range(num_islands):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 15: # Need at least 10s for LS + 5s margin
            break
        
        # Generate initialization
        # Add some randomness to the spectral partition to diversify
        if k == 0:
            p_init = _spectral_partition_init(a, b, seed=42)
        else:
            # Perturb the spectral init
            p_init = _spectral_partition_init(a, b, seed=42 + k)
            rng = random.Random(100 + k)
            num_swaps = n // 8
            for _ in range(num_swaps):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
        
        # Run local search
        # Allocate remaining time / remaining islands
        time_per_ls = (remaining_time - 5) / (num_islands - k)
        
        p_best, cost_best = _local_search_2opt(a, b, p_init, time_per_ls, start_time, seed=1000 + k)
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        # Fallback
        best_p = _spectral_partition_init(a, b, seed=0)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
