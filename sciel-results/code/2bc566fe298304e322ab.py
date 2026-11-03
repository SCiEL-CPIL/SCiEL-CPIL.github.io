"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _generate_all_vectors():
    """
    Generate a diverse pool of candidate vectors.
    We focus on shells with small norms to maximize packing density.
    Shell 1: 11 vectors (norm 1)
    Shell 2: 220 vectors (norm sqrt(2))
    Shell 3: 1320 vectors (norm sqrt(3))
    Shell 4: 5280 vectors (norm sqrt(4) = 2)
    
    Total ~ 6831 vectors.
    """
    vecs = []
    
    # Shell 1: e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2: e_i - e_j
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3: Weight 3. C(11,3) * 2^3 = 165 * 8 = 1320.
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4: Weight 4. C(11,4) * 2^4 = 330 * 16 = 5280.
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def _build_conflict_graph(candidates, norms_sq):
    """
    Build the conflict graph adjacency matrix.
    Two vectors u, v conflict if dist_sq(u, v) < max(norm_sq_u, norm_sq_v).
    """
    n = len(candidates)
    # We need to check all pairs. n is ~7000. n^2 is 49M. 
    # Computing dot products for all pairs is n^2 * D = 49M * 11 = 539M ops. 
    # This might be too slow in pure Python/NumPy if not careful, but NumPy vectorization helps.
    # However, we can be smarter. We only care about conflicts.
    # Let's compute the Gram matrix G = candidates @ candidates.T.
    # This is n x n. 7000x7000 = 49M floats. ~400MB. Might be tight but okay for 120s?
    # Actually, 49M * 8 bytes = 392MB. Python overhead is high. 
    # Let's try to avoid the full matrix if possible, or use a block-based approach.
    # Or, we can use the fact that most vectors are sparse.
    # But let's try the direct approach first with a time check.
    
    # To save memory and time, we can compute conflicts in blocks.
    # Or, we can just use a list of neighbors for MIS.
    # For MIS, we need to know neighbors quickly.
    # Let's build an adjacency list.
    
    neighbors = [[] for _ in range(n)]
    
    # Precompute norms
    # dist_sq(i, j) = norms_sq[i] + norms_sq[j] - 2 * dot(i, j)
    # Conflict if dist_sq < max(norms_sq[i], norms_sq[j])
    
    # Block size to keep memory manageable and allow time checks
    block_size = 500
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = candidates[start:end]
        block_norms = norms_sq[start:end]
        
        # Compute dots between block and all others
        # block @ candidates.T -> (block_size, n)
        dots = block @ candidates.T
        
        # dists_sq = block_norms[:, None] + norms_sq[None, :] - 2 * dots
        # This creates a (block_size, n) matrix.
        dists_sq = block_norms[:, None] + norms_sq[None, :] - 2 * dots
        
        # reqs = np.maximum(block_norms[:, None], norms_sq[None, :])
        reqs = np.maximum(block_norms[:, None], norms_sq[None, :])
        
        # conflict_mask = dists_sq < reqs
        conflict_mask = dists_sq < reqs
        
        # Exclude self (diagonal is 0 < norm, so it's True, but we don't want self-loops)
        # The diagonal elements correspond to i==j.
        # For i in [start, end), j=i is at column i.
        # We can just ignore the diagonal when adding to neighbors.
        
        for local_i in range(end - start):
            global_i = start + local_i
            # Find conflicts in this block row
            conflicts = np.where(conflict_mask[local_i])[0]
            for j in conflicts:
                if global_i != j:
                    neighbors[global_i].append(j)
                    
        if time.time() > 140: # Safety margin
             break
             
    return neighbors

def _tomita_mis(neighbors, n, deadline):
    """
    Implement Bron-Kerbosch with pivot (Tomita's algorithm) for Maximum Independent Set.
    We want the largest set of vertices with no edges between them.
    This is equivalent to Maximum Clique in the complement graph.
    Or, we can run it directly on the "compatibility" graph.
    Let's define compatibility graph: edge if NOT conflicting.
    Then we want Max Clique in Compatibility Graph.
    Or Max Independent Set in Conflict Graph.
    
    Bron-Kerbosch finds Max Clique.
    Let's work with the Conflict Graph neighbors.
    We want a set S such that no two nodes in S are connected in `neighbors`.
    This is an Independent Set.
    
    Standard Bron-Kerbosch finds Cliques.
    To find Independent Sets, we can run it on the Complement Graph.
    But building the complement is expensive.
    
    Alternatively, we can modify Bron-Kerbosch to find Independent Sets.
    R: current independent set
    P: candidates that can be added to R (must not be connected to any node in R)
    X: candidates that were removed but cannot be added (connected to some node in R? No, that's for Clique).
    
    Let's stick to finding Max Clique in the Compatibility Graph.
    Compatibility(i, j) = (i, j) NOT in neighbors[i].
    
    Actually, it's easier to just implement a greedy MIS with a good heuristic, 
    or use a library if available? No, standard library only.
    
    Let's implement a simple but effective heuristic: 
    1. Sort vertices by degree (ascending).
    2. Greedily pick the vertex with lowest degree, add to set, remove it and its neighbors.
    3. Repeat.
    
    This is O(n * m). With n=7000, m~n^2/2, this is too slow.
    
    Let's try a "Local Search" for MIS which is faster and often good enough.
    But the prompt asks for a DIFFERENT strategy. 
    The current program uses random swaps.
    
    Let's try **Greedy MIS with Degree Ordering** first, then **Local Search (1-swap, 2-swap)** to improve.
    
    """
    # 1. Greedy MIS
    # Degree of each node
    degrees = [len(neighbors[i]) for i in range(n)]
    
    # Order by degree
    order = sorted(range(n), key=lambda x: degrees[x])
    
    selected = set()
    removed = set()
    
    for i in order:
        if i in removed:
            continue
        if i in selected:
            continue
            
        # Check if i is compatible with all currently selected
        # i is compatible with S if no neighbor of i is in S.
        is_compat = True
        for nb in neighbors[i]:
            if nb in selected:
                is_compat = False
                break
                
        if is_compat:
            selected.add(i)
            # Remove i and all its neighbors from consideration
            removed.add(i)
            for nb in neighbors[i]:
                removed.add(nb)
                
    current_set = list(selected)
    current_size = len(current_set)
    
    # 2. Local Search: Try to swap one node out and two in? 
    # Or just 1-for-1 swaps to "anneal" the set?
    # Since we want to maximize size, 1-for-1 doesn't help directly unless it enables future additions.
    # A better move: Try to add a node. If it conflicts, try to remove one conflict.
    # This is what the current program does.
    
    # Let's try a "2-for-1" move: Remove 1 node, Add 2 nodes?
    # This is complex to search.
    
    # Let's try **Simulated Annealing** on the MIS size.
    # State: Set of indices.
    # Move: Randomly remove a node, Randomly add a compatible node?
    # If we remove a node, we might be able to add 2 or 3 others.
    
    # Let's implement a simple "Greedy + Random Restart" with different orderings.
    # And a "Hill Climbing" that tries to swap a node with a random non-neighbor.
    
    # Actually, the most impactful change is to use the **structure** of the vectors.
    # Vectors with similar directions are more likely to conflict.
    
    # Let's try **Spectral Greedy**:
    # 1. Compute Laplacian of the Conflict Graph? Too expensive.
    # 2. Use PCA on the vectors to group them.
    
    # Let's stick to a robust **Greedy MIS with Randomized Degree Ordering** and **Local Search**.
    
    # Local Search: 
    # While time remains:
    #   Pick a random node v NOT in set.
    #   Find its conflicts in the set.
    #   If no conflicts, add it.
    #   If conflicts exist, try to remove one conflict c.
    #   Check if v is compatible with Set \ {c}. (It is, by definition of conflict).
    #   But does removing c allow adding v AND another node w?
    #   Let's just do 1-for-1 swaps to explore the space, hoping to find a "hole" later.
    
    # To make it better, let's try **2-for-1**:
    # Remove c. Can we add v and some other w?
    # This is expensive.
    
    # Let's just do a fast **Greedy** with multiple random restarts and pick the best.
    # And then a **Local Search** that tries to add nodes by removing *all* their conflicts?
    # If v has 2 conflicts, remove both, add v. Size change: -2 + 1 = -1. Bad.
    # So only add if conflicts == 0.
    # Swap if conflicts == 1.
    
    # The current program does this. What is materially different?
    # The current program starts with a greedy sort by norm.
    # My program starts with a greedy sort by **Degree in Conflict Graph**.
    # This is a structural change in the initialization heuristic.
    
    # Let's refine the Local Search to be more aggressive.
    # Instead of random 1-for-1, try **2-for-2**?
    
    # Let's implement a **Tabu Search** or **Iterated Local Search**?
    
    # Given the time, let's implement **Greedy by Degree** + **Randomized Local Search** 
    # with a focus on **adding nodes that have low degree in the remaining graph**.
    
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s buffer
    
    # 1. Generate candidates
    candidates = _generate_all_vectors()
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Build Conflict Graph
    # This is the most expensive part.
    neighbors = _build_conflict_graph(candidates, norms_sq)
    
    # 3. Solve MIS
    best_set = []
    
    # Try multiple random restarts of Greedy MIS
    for restart in range(10):
        if time.time() > deadline - 10: break
        
        # Randomize the order slightly
        # Or just use the degree order
        degrees = [len(neighbors[i]) for i in range(n_cand)]
        # Add small random noise to degree to break ties differently
        noisy_degrees = [degrees[i] + np.random.rand() for i in range(n_cand)]
        order = sorted(range(n_cand), key=lambda x: noisy_degrees[x])
        
        selected = set()
        removed = set()
        
        for i in order:
            if i in removed:
                continue
            if i in selected:
                continue
                
            # Check compatibility
            is_compat = True
            for nb in neighbors[i]:
                if nb in selected:
                    is_compat = False
                    break
                    
            if is_compat:
                selected.add(i)
                removed.add(i)
                for nb in neighbors[i]:
                    removed.add(nb)
                    
        if len(selected) > len(best_set):
            best_set = list(selected)
            
        # Local Search on this set
        current_set = set(best_set)
        # We need to be able to check compatibility quickly.
        # neighbors is a list of lists.
        
        # To speed up local search, we can maintain a "blocked" set for each candidate?
        # No, just check against current_set.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add a random node
            # We can iterate through all nodes not in set? n_cand is 7000.
            # For each node, check if compatible.
            # If compatible, add it.
            # This is O(n_cand * avg_degree).
            
            # To be faster, let's just pick random nodes.
            for _ in range(1000):
                if time.time() > deadline - 5: break
                v = np.random.randint(0, n_cand)
                if v in current_set:
                    continue
                    
                # Check conflicts
                conflicts = [nb for nb in neighbors[v] if nb in current_set]
                
                if len(conflicts) == 0:
                    current_set.add(v)
                    improved = True
                elif len(conflicts) == 1:
                    # Try 1-for-1 swap? 
                    # This doesn't increase size, but might help.
                    # Let's skip 1-for-1 for now to focus on size increase.
                    pass
                else:
                    # Try 2-for-1? No.
                    pass
                    
            if len(current_set) > len(best_set):
                best_set = list(current_set)
                
    # Final Result
    if not best_set:
        return np.array([np.zeros(D, dtype=np.int64) + 1], dtype=np.int64)
        
    return candidates[best_set]

# EVOLVE-BLOCK-END
