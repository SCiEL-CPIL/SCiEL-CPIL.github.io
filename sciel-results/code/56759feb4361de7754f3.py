import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert supports to bitmasks for fast intersection check
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build compatibility graph (compatibility = intersection size <= 2)
    # This is the complement of the conflict graph (intersection size == 3)
    # We want to find a Maximum Clique in this compatibility graph.
    # Edges exist if NOT (intersection == 3)
    
    # adj_compat[i][j] is True if supports[i] and supports[j] are compatible
    adj_compat = np.zeros((N_S, N_S), dtype=bool)
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter != 3:
                adj_compat[i][j] = True
                adj_compat[j][i] = True

    # 3. Maximum Clique via Branch and Bound with Tomita Coloring
    
    # We represent the candidate set P as a list of vertices.
    # We use a recursive function that tries to add vertices to the clique C.
    # To speed up, we use coloring on the induced subgraph of P to get an upper bound.
    
    # Precompute adjacency lists for faster access in Python loops?
    # Numpy boolean indexing is fast for 330x330.
    
    # Tomita's algorithm (MCQ) typically uses a specific coloring heuristic.
    # Simplified version:
    # 1. Order vertices in P.
    # 2. Greedily color the induced subgraph of P.
    # 3. The color class count is the upper bound on clique size in P.
    # 4. Branch on vertices in reverse order of coloring (highest color first).
    
    # For 330 nodes, a pure Python implementation of Tomita might be slow if not careful.
    # However, 330 is small enough that a well-optimized Python recursion with numpy
    # for neighbor checks might work, or we can stick to the independent set approach
    # but with a better heuristic.
    
    # Let's try the Independent Set approach on the Conflict Graph (intersection == 3)
    # because the conflict graph is likely sparser than the compatibility graph.
    # Conflict graph edges: intersection == 3.
    # Max Independent Set in Conflict Graph == Max Clique in Compatibility Graph.
    
    adj_conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj_conflict[i][j] = True
                adj_conflict[j][i] = True

    # Heuristic: Max Independent Set
    # We use a greedy + local search approach, but we will enhance it with a 
    # "branch and bound" style improvement or a more robust local search.
    
    # Given the time limit and the size (330), a simple greedy + local search 
    # is often competitive. The inspirations used similar structures.
    # Let's try to implement a slightly better local search that allows 
    # "2-move" swaps more effectively or uses a different initialization.
    
    # Alternative: Use the structure of the problem.
    # Supports are 4-subsets of {0..10}.
    # Two supports are incompatible if they share exactly 3 elements.
    # This means they differ by exactly 1 element.
    # Let S = {a,b,c,d}. Incompatible with {a,b,c,e} for any e not in S.
    
    # Let's stick to the general graph algorithm but optimize the local search.
    
    best_size = 0
    best_set_mask = np.zeros(N_S, dtype=bool)
    rng = np.random.RandomState(42)

    def get_size(mask):
        return int(np.sum(mask))

    def greedy_random():
        """Randomized greedy independent set."""
        order = rng.permutation(N_S)
        selected = np.zeros(N_S, dtype=bool)
        # Precompute degrees? Or just use the mask.
        for v in order:
            if not selected[v]:
                if not np.any(selected & adj_conflict[v]):
                    selected[v] = True
        return selected

    def local_search(init_mask, max_iter=500):
        """
        Local search on boolean mask.
        Tries to add nodes. If stuck, performs 1-swap or 2-swap.
        """
        current = init_mask.copy()
        best_local = current.copy()
        best_local_size = get_size(best_local)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute blocked nodes (neighbors of current set)
            if np.any(current):
                blocked = np.any(adj_conflict[current], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            # Candidates: not in current, not blocked
            candidates = ~(current | blocked)
            cand_indices = np.where(candidates)[0]
            
            if len(cand_indices) > 0:
                # Greedily add one
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current[v_add] = True
                
                if get_size(current) > best_local_size:
                    best_local_size = get_size(current)
                    best_local = current.copy()
                continue
            
            # Stuck: No candidates. Try 1-swap.
            current_indices = np.where(current)[0]
            if len(current_indices) == 0:
                break
                
            improved = False
            
            # Try 1-swap
            # To improve efficiency, we can pre-compute potential swaps?
            # For now, random sampling.
            for _ in range(20): # Increased attempts
                if len(current_indices) == 0:
                    break
                v_remove = int(current_indices[rng.randint(len(current_indices))])
                current[v_remove] = False
                
                # Recompute blocked
                if np.any(current):
                    blocked2 = np.any(adj_conflict[current], axis=0)
                else:
                    blocked2 = np.zeros(N_S, dtype=bool)
                
                candidates2 = ~(current | blocked2)
                cand_indices2 = np.where(candidates2)[0]
                
                if len(cand_indices2) > 0:
                    # Try to add a candidate
                    v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                    current[v_add] = True
                    
                    new_size = get_size(current)
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = current.copy()
                        improved = True
                        current_indices = np.where(current)[0]
                        break
                    else:
                        # Revert swap if no improvement
                        current[v_add] = False
                        current[v_remove] = True
                        current_indices = np.where(current)[0]
            
            if improved:
                continue
            
            # Try 2-swap
            if len(current_indices) >= 2:
                # Pick two distinct nodes to remove
                i1, i2 = rng.choice(len(current_indices), replace=False, size=2)
                v_rem1 = int(current_indices[i1])
                v_rem2 = int(current_indices[i2])
                
                current[v_rem1] = False
                current[v_rem2] = False
                
                # Recompute blocked
                if np.any(current):
                    blocked3 = np.any(adj_conflict[current], axis=0)
                else:
                    blocked3 = np.zeros(N_S, dtype=bool)
                
                candidates3 = ~(current | blocked3)
                cand_indices3 = np.where(candidates3)[0]
                
                # We want to add at least 2 nodes to improve size
                added = []
                temp_current = current.copy()
                
                # Try to add 2
                for _ in range(2):
                    if len(cand_indices3) == 0:
                        break
                    valid_cands = []
                    for c in cand_indices3:
                        if not np.any(temp_current & adj_conflict[c]):
                            valid_cands.append(c)
                    
                    if not valid_cands:
                        break
                        
                    v_add = int(valid_cands[rng.randint(len(valid_cands))])
                    temp_current[v_add] = True
                    added.append(v_add)
                
                if len(added) >= 2:
                    for v in added:
                        current[v] = True
                    
                    new_size = get_size(current)
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = current.copy()
                        improved = True
                        current_indices = np.where(current)[0]
                
                if not improved:
                    # Revert 2-swap
                    for v in added:
                        current[v] = False
                    current[v_rem1] = True
                    current[v_rem2] = True
                    current_indices = np.where(current)[0]
            
            if not improved:
                # Try a "kick" or random restart of local search?
                # Or just break and let outer loop restart.
                break

        return best_local

    # Main loop
    # Increase restarts if possible, but keep within deadline.
    # We can also try different initialization strategies.
    
    num_restarts = 3000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Mix of random greedy and min-degree greedy?
        # Min-degree greedy often finds larger independent sets.
        if r % 2 == 0:
            init_mask = greedy_random()
        else:
            # Simple min-degree heuristic
            # Select vertex with min degree in remaining graph, add to set, remove it and neighbors.
            # This is O(N^2) per step, N=330. 330^2 * 330 steps? Too slow in Python loop.
            # Let's stick to random greedy but with different permutations.
            init_mask = greedy_random()
        
        final_mask = local_search(init_mask, max_iter=200)
        
        size = get_size(final_mask)
        if size > best_size:
            best_size = size
            best_set_mask = final_mask.copy()

    # 4. Construct vectors
    # For each support in best_set_mask, add all 16 sign vectors
    vectors = []
    for s_idx in range(N_S):
        if best_set_mask[s_idx]:
            support = supports[s_idx]
            # Generate all 16 sign combinations
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int8)
                for idx, sign in zip(support, signs):
                    v[idx] = sign
                vectors.append(v)
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
