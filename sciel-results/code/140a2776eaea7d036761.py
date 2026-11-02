"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def _generate_candidates(max_norm_sq=8):
    """
    Generate all nonzero integer vectors in R^D with squared norm <= max_norm_sq.
    We include norms 1, 2, 3, 4, 5, 6, 7, 8 to have a dense pool.
    """
    candidates = []
    
    # Helper to add a vector
    def add_vec(v):
        # Check if it's a duplicate (up to sign? No, v and -v are distinct vectors 
        # but they might be incompatible with each other? 
        # ||v - (-v)||^2 = 4||v||^2 >= max(||v||^2, ||v||^2) = ||v||^2. 
        # So v and -v are ALWAYS compatible. 
        # We must include both to maximize count.
        candidates.append(v)

    # Iterate over possible squared norms
    for n_sq in range(1, max_norm_sq + 1):
        # Generate all integer vectors v in Z^D such that ||v||^2 = n_sq
        # This is a partition problem. 
        # For small D=11 and small n_sq, we can use recursion or itertools.
        
        # Efficient generation for small norms:
        # We can iterate over all possible tuples of integers? 
        # Range of entries: if n_sq=8, max entry is 2 (sqrt(8)). 
        # Actually, max entry is floor(sqrt(n_sq)).
        
        max_entry = int(np.sqrt(n_sq))
        
        # Recursively generate vectors
        def gen(prefix, remaining_sq, depth):
            if depth == D:
                if remaining_sq == 0:
                    if np.any(prefix): # Non-zero
                        add_vec(np.array(prefix, dtype=np.int64))
                return
            
            # Possible values for current coordinate
            # v^2 <= remaining_sq
            max_val = int(np.sqrt(remaining_sq))
            for val in range(-max_val, max_val + 1):
                new_rem = remaining_sq - val*val
                if new_rem >= 0:
                    gen(prefix + [val], new_rem, depth + 1)
        
        gen([], n_sq, 0)
        
    return np.array(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    # 1. Generate Candidate Pool
    # Norms up to 8 provide a very dense set.
    # Count estimate:
    # Norm 1: 22
    # Norm 2: 220
    # Norm 3: 1980
    # Norm 4: ~5500
    # Norm 5: ~20000 (e.g., 1^2+2^2)
    # Norm 6: ~?
    # Norm 7: ~?
    # Norm 8: ~?
    # Total might be ~50,000 - 100,000 vectors.
    # Building a graph of 100k nodes is feasible in memory if sparse.
    
    candidates = _generate_candidates(max_norm_sq=8)
    n_cand = len(candidates)
    
    if n_cand == 0:
        return np.array([[1] + [0]*(D-1)], dtype=np.int64)

    norms_sq = np.sum(candidates * candidates, axis=1)

    # 2. Build Conflict Graph
    # Node i and j are connected if they are INCOMPATIBLE.
    # Condition for incompatibility: 2 * v_i . v_j > min(||v_i||^2, ||v_j||^2)
    # We want to find a Maximum Independent Set (MIS) in this graph.
    # MIS in general graphs is NP-hard. However, for geometric packing, the graph might have structure.
    # Alternatively, we can use a greedy heuristic on the graph, or a local search.
    # Given the size (~50k-100k), exact MIS is impossible.
    # We will use a fast greedy algorithm with randomization and restarts, 
    # but optimized using the graph structure.
    
    # Pre-compute the conflict matrix? 
    # 100k x 100k boolean matrix is 10^10 bits = 1.25 GB. Too big for RAM usually.
    # We need a streaming or block-based approach.
    
    # Strategy: 
    # Use a "Greedy with Backtracking" or "Local Search" directly on vectors.
    # To make it fast, we use a "Conflict List" approach.
    
    # Let's try a different structural approach: 
    # **Max-Flow / Min-Cut approximation?** No.
    # **Local Search (Simulated Annealing) is what the parent does.**
    # **What is different?**
    # The parent uses a sparse pool (norms 1-4) and random restarts.
    # I will use a **denser pool** (norms 1-8) and a **cluster-based greedy** approach.
    # 
    # Insight: Vectors with the same norm and similar directions are likely to conflict.
    # We can partition candidates into "buckets" by norm.
    # Within a norm, the problem is equivalent to finding a set of points on a sphere with minimum angular separation.
    
    # Let's implement a **fast greedy with priority queue** based on "degree of freedom" or "potential".
    # Actually, the most effective heuristic for packing in high dimensions is often **random greedy with a large pool**.
    # The key improvement from the parent:
    # 1. Larger pool (norms up to 8).
    # 2. Better ordering: Sort candidates by norm (smaller norms are "harder" to pack? No, smaller norms have smaller "footprint" in distance space? 
    #    Condition: 2 v.w <= min(n_v, n_w).
    #    If n_v is small, the limit is tight. So small norm vectors are more restrictive.
    #    Therefore, we should pick small norm vectors FIRST? 
    #    Or does picking large norm vectors first allow more small ones?
    #    Let's test both orderings via random restarts.
    
    # To speed up compatibility checks:
    # We maintain a list of selected vectors.
    # Checking a new vector against K selected vectors is O(K*D).
    # With K~500, D=11, this is very fast.
    
    best_set = []
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Pre-shuffle candidates for random restarts
    # We will do multiple restarts with different orderings.
    
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Strategy: 
        # 1. Sort by norm (ascending or descending) with some noise?
        # 2. Or just random permutation.
        
        # Let's try sorting by norm ascending. Small norms are "tighter".
        # If we pick a small norm vector, it forbids many others.
        # If we pick a large norm vector, it forbids fewer?
        # Let's check: 
        # v_small (norm 1), w (norm 4). Limit: min(1, 4) = 1. 2 v.w <= 1 => v.w <= 0.5.
        # v_large (norm 4), w (norm 1). Limit: min(4, 1) = 1. 2 v.w <= 1 => v.w <= 0.5.
        # Symmetric.
        # However, the "angle" constraint is tighter for the smaller norm relative to its own magnitude.
        # cos(theta) <= 0.5 * min(n_v, n_w) / (||v|| ||w||)
        # If n_v < n_w, cos(theta) <= 0.5 * n_v / (sqrt(n_v) sqrt(n_w)) = 0.5 * sqrt(n_v/n_w).
        # As n_w increases, the allowed angle increases (looser constraint).
        # So large norm vectors are "easier" to pack with small ones?
        # Actually, it's symmetric in the pair.
        
        # Let's use a "Conflict-Aware" Greedy:
        # At each step, pick the candidate that conflicts with the FEWEST already selected vectors?
        # That's expensive to compute dynamically.
        
        # Let's stick to Random Restart Greedy but with a much larger pool.
        # The parent's bottleneck might be the small pool size.
        
        perm = rng.permutation(n_cand)
        curr_set = []
        curr_norms = []
        
        # Optimization: 
        # Instead of checking against all curr_set, we can just do it.
        # With n_cand ~ 50k and K ~ 500, 
        # Total checks: 50k * 500 * 11 ~ 275 million flops.
        # In NumPy, this is vectorized per candidate.
        # 50k iterations. Each iteration does a dot product of (K, D) @ (D,) -> (K,).
        # This is fast.
        
        for idx in perm:
            v = candidates[idx]
            n_v = norms_sq[idx]
            
            if not curr_set:
                curr_set.append(v)
                curr_norms.append(n_v)
                continue
                
            sel_arr = np.array(curr_set)
            sel_norms_arr = np.array(curr_norms)
            
            dots = sel_arr @ v
            limits = np.minimum(n_v, sel_norms_arr)
            
            if np.all(2 * dots <= limits):
                curr_set.append(v)
                curr_norms.append(n_v)
                
        if len(curr_set) > best_count:
            best_count = len(curr_set)
            best_set = curr_set.copy()
            
    # 3. Local Search Refinement
    # Try to swap vectors to improve the count.
    # We use the best set found.
    curr_set = best_set
    curr_norms = best_norms = best_count # wait, best_norms not stored correctly in loop above?
    # Let's re-store best_norms properly.
    
    # Re-run best restart to get norms
    # Actually, let's just keep the set and recompute norms if needed, or store them.
    # In the loop above, I didn't store best_norms. Let's fix that.
    
    # Reset
    best_set = []
    best_norms = []
    best_count = 0
    
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        perm = rng.permutation(n_cand)
        curr_set = []
        curr_norms = []
        
        for idx in perm:
            v = candidates[idx]
            n_v = norms_sq[idx]
            
            if not curr_set:
                curr_set.append(v)
                curr_norms.append(n_v)
                continue
                
            sel_arr = np.array(curr_set)
            sel_norms_arr = np.array(curr_norms)
            
            dots = sel_arr @ v
            limits = np.minimum(n_v, sel_norms_arr)
            
            if np.all(2 * dots <= limits):
                curr_set.append(v)
                curr_norms.append(n_v)
                
        if len(curr_set) > best_count:
            best_count = len(curr_set)
            best_set = curr_set.copy()
            best_norms = curr_norms.copy()
            
    # Simple SA on the best set
    current_set = best_set
    current_norms = best_norms
    current_count = len(current_set)
    
    T = 1.0
    T_min = 1e-4
    alpha = 0.999
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        idx_new = rng.integers(0, n_cand)
        v_new = candidates[idx_new]
        n_new = norms_sq[idx_new]
        
        if not current_set:
            current_set.append(v_new)
            current_norms.append(n_new)
            continue
            
        sel_arr = np.array(current_set)
        sel_norms_arr = np.array(current_norms)
        
        dots = sel_arr @ v_new
        limits = np.minimum(n_new, sel_norms_arr)
        conflicts = 2 * dots > limits
        
        if not np.any(conflicts):
            current_set.append(v_new)
            current_norms.append(n_new)
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set.copy()
                best_norms = current_norms.copy()
        else:
            conflict_indices = np.where(conflicts)[0]
            idx_conf = conflict_indices[rng.integers(0, len(conflict_indices))]
            
            v_old = current_set[idx_conf]
            n_old = current_norms[idx_conf]
            
            current_set.pop(idx_conf)
            current_norms.pop(idx_conf)
            
            if current_set:
                sel_arr_rest = np.array(current_set)
                sel_norms_rest = np.array(current_norms)
                dots_rest = sel_arr_rest @ v_new
                limits_rest = np.minimum(n_new, sel_norms_rest)
                if np.all(2 * dots_rest <= limits_rest):
                    current_set.append(v_new)
                    current_norms.append(n_new)
                else:
                    current_set.append(v_old)
                    current_norms.append(n_old)
            else:
                current_set.append(v_new)
                current_norms.append(n_new)
                
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set.copy()
            best_norms = current_norms.copy()
            
        T *= alpha
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
