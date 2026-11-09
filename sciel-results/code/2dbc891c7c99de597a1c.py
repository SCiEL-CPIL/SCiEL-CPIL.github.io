import math
import time

def construct():
    """
    Returns a maximal pairwise noncoprime subset of [1, 255] containing 255.
    
    Strategy:
    1. Identify all candidates in [1, 255] that share a factor with 255 (3, 5, or 17).
    2. Start with the set of all multiples of 3 (size 85), which is a valid clique.
    3. Perform a local search to see if we can increase the size by swapping elements.
       - Try to add any compatible element.
       - Try to remove one element and add multiple compatible elements (1-for-k swaps).
    4. Return the best set found.
    """
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    # 1. Candidates
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    cand_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = cand_idx[255]
    
    # 2. Precompute incompatibility (gcd == 1)
    # incompat[i][j] is True if candidates[i] and candidates[j] are coprime
    incompat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) == 1:
                incompat[i][j] = True
                incompat[j][i] = True
    
    # Helper: check if node x is compatible with all nodes in set S (indices)
    def is_compatible_with_all(x, S):
        for y in S:
            if incompat[x][y]:
                return False
        return True
    
    # Helper: greedily find a maximal clique within a set of candidates compatible with a base set
    # This is used for the "add" part of swaps.
    def greedy_max_clique(base_set, available_nodes):
        # available_nodes: list of indices compatible with all in base_set
        # We want to find a large clique among available_nodes
        current_clique = set()
        # Sort by degree in the available subgraph to prioritize high-connectivity nodes?
        # Or just iterate.
        # For small sizes, simple greedy is fine.
        remaining = list(available_nodes)
        # Heuristic: pick node with highest degree in the remaining subgraph
        while remaining:
            # Calculate degrees within remaining
            best_node = None
            best_deg = -1
            for node in remaining:
                deg = sum(1 for other in remaining if not incompat[node][other])
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
            
            if best_node is None:
                break
                
            # Check if best_node is compatible with current_clique
            if is_compatible_with_all(best_node, current_clique):
                current_clique.add(best_node)
                remaining.remove(best_node)
            else:
                # If not compatible with current clique, remove it?
                # Actually, if it's not compatible with current_clique, it can't be added.
                # But maybe another node is.
                # To avoid infinite loop if best_node is bad, we just remove it from consideration for this step?
                # No, we should try to build the clique.
                # If best_node is incompatible with current_clique, we cannot add it.
                # We should remove it from 'remaining' to try the next best?
                # Or just break?
                # Let's just remove it.
                remaining.remove(best_node)
                
        return current_clique

    # 3. Initialize with multiples of 3
    current_set = set(i for i, v in enumerate(candidates) if v % 3 == 0)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Ensure 255 is in best_set (it is, 255 % 3 == 0)
    
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        current_size = len(current_set)
        
        # Phase 1: Try single additions
        for x in range(n):
            if x in current_set:
                continue
            if is_compatible_with_all(x, current_set):
                current_set.add(x)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try 1-for-k swaps
            # For each node u in current_set (excluding 255), try removing it
            # and adding a clique from the nodes that become compatible.
            found_swap = False
            for u in list(current_set):
                if u == idx_255:
                    continue
                
                test_set = current_set - {u}
                
                # Find all nodes not in test_set that are compatible with all in test_set
                # These are candidates for addition
                add_candidates = []
                for v in range(n):
                    if v in test_set:
                        continue
                    if is_compatible_with_all(v, test_set):
                        add_candidates.append(v)
                
                # Find a maximal clique among add_candidates
                # We want to add as many as possible
                new_nodes = greedy_max_clique(test_set, add_candidates)
                
                if len(new_nodes) > 1: # If we can add more than 1, it's a net gain
                    # Verify the new set is valid (it should be by construction)
                    new_set = test_set | new_nodes
                    if len(new_set) > current_size:
                        current_set = new_set
                        current_size = len(new_set)
                        improved = True
                        found_swap = True
                        break
            
            # If no 1-for-k found, try 2-for-k?
            # This is expensive. Given the structure, 1-for-k is likely sufficient to find improvements if they exist near the local optimum.
            # If we are stuck at 85, and no 1-for-k works, we are likely at a local maximum.
            # Given the strong theoretical evidence that 85 is global max, this is fine.

        if current_size > best_size:
            best_set = set(current_set)
            best_size = current_size

    return [candidates[i] for i in best_set]
