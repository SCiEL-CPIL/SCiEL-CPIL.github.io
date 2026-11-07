import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute validity of all subsets
    # is_valid_part[mask] is True if mask is a clique or an independent set
    is_valid_part = [False] * (1 << n)
    
    def is_clique(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        # All other vertices in mask must be adjacent to v
        others = mask ^ (1 << v)
        if others & ~adj_mask[v]:
            return False
        return is_clique(others)
        
    def is_independent(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        # No other vertices in mask can be adjacent to v
        others = mask ^ (1 << v)
        if others & adj_mask[v]:
            return False
        return is_independent(others)

    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            is_valid_part[mask] = True
            
    # Greedy upper bound
    def greedy_partition():
        parts = []
        remaining = (1 << n) - 1
        while remaining:
            # Find a maximal valid subset of remaining
            # Simple heuristic: start with lowest vertex, add others if valid
            # This is O(n^2 * 2^n) worst case if done naively, but n=14 is small.
            # Better: just pick a vertex and find a large valid set containing it.
            # For upper bound, any valid partition works.
            # Let's try to build one part at a time.
            v = (remaining & -remaining).bit_length() - 1
            part_mask = 1 << v
            # Try to add other vertices to this part
            # Iterate through remaining vertices
            # To make it fast, we can just try to extend greedily
            # Check if adding u to part_mask keeps it valid
            # We need to check clique or independent property.
            # Since we precomputed is_valid_part, we can just check is_valid_part[part_mask | (1<<u)]
            # But we need to know if it's a clique or independent set to extend correctly?
            # No, is_valid_part just tells if the *result* is valid.
            # So we can try to add vertices one by one.
            for u in range(n):
                if u == v: continue
                if remaining & (1 << u):
                    new_mask = part_mask | (1 << u)
                    if is_valid_part[new_mask]:
                        part_mask = new_mask
            parts.append(part_mask)
            remaining ^= part_mask
        return parts

    greedy_parts = greedy_partition()
    best_k = len(greedy_parts)
    best_partition_masks = greedy_parts[:]
    
    # Iterative Deepening DFS
    # State: (remaining_mask, k_left)
    # Returns True if remaining_mask can be partitioned into <= k_left valid parts.
    # We store the partition found if successful.
    
    deadline = time.time() + 85.0
    found_solution = False
    result_partition = None
    
    # Memoization: set of (remaining_mask, k_left) that failed
    failed_states = set()
    
    def dfs(remaining, k_left):
        nonlocal deadline, found_solution, result_partition
        
        if time.time() > deadline:
            return False
            
        if remaining == 0:
            return True
            
        if k_left == 0:
            return False
            
        if is_valid_part[remaining]:
            return True
            
        state = (remaining, k_left)
        if state in failed_states:
            return False
            
        # Pick the lowest set bit vertex to break symmetry
        # This vertex MUST be in the first part we form in this recursion level
        # We iterate over all valid subsets P of 'remaining' that contain this vertex
        # and check if 'remaining \ P' can be partitioned with k_left - 1 parts.
        
        v_bit = remaining & -remaining
        v_idx = v_bit.bit_length() - 1
        
        # Generate all subsets of 'remaining' that contain v_bit
        # We can iterate through all subsets of (remaining ^ v_bit) and add v_bit
        rest = remaining ^ v_bit
        sub = rest
        while True:
            candidate_part = sub | v_bit
            if is_valid_part[candidate_part]:
                new_remaining = remaining ^ candidate_part
                if dfs(new_remaining, k_left - 1):
                    # Success! We need to reconstruct the partition.
                    # This requires storing the path. 
                    # For n=14, we can just return the partition as a list of masks.
                    # Let's modify dfs to return the list of masks if successful.
                    pass 
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        failed_states.add(state)
        return False
        
    # The above DFS doesn't return the partition. Let's rewrite to return the partition.
    # dfs(remaining, k_left) -> list of masks or None
    
    def dfs_with_result(remaining, k_left):
        nonlocal deadline
        
        if time.time() > deadline:
            return None
            
        if remaining == 0:
            return []
            
        if k_left == 0:
            return None
            
        if is_valid_part[remaining]:
            return [remaining]
            
        state = (remaining, k_left)
        if state in failed_states:
            return None
            
        v_bit = remaining & -remaining
        rest = remaining ^ v_bit
        sub = rest
        while True:
            candidate_part = sub | v_bit
            if is_valid_part[candidate_part]:
                new_remaining = remaining ^ candidate_part
                res = dfs_with_result(new_remaining, k_left - 1)
                if res is not None:
                    return [candidate_part] + res
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
        failed_states.add(state)
        return None

    # Try k from 1 to best_k
    for k in range(1, best_k + 1):
        if time.time() > deadline:
            break
        failed_states.clear() # Reset memoization for new k? 
        # Actually, if (S, k') failed for k' > k, it doesn't imply (S, k) fails.
        # But if (S, k) failed, it's useless for larger k in the same search?
        # No, we are searching for a specific k. 
        # However, if we search k=1, then k=2, the failed states from k=1 are not directly applicable to k=2 
        # because the "k_left" dimension is different.
        # So we should clear failed_states for each new k iteration to be safe, 
        # or include k in the state key (which we did: state=(remaining, k_left)).
        # So we don't need to clear it if the state includes k_left.
        
        res_masks = dfs_with_result((1 << n) - 1, k)
        if res_masks is not None:
            best_k = k
            best_partition_masks = res_masks
            break # Found minimal k
            
    # Convert best_partition_masks to list of lists of ints
    final_result = []
    for mask in best_partition_masks:
        verts = []
        temp = mask
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        # Ensure sorted for determinism, though not strictly required
        verts.sort()
        final_result.append(verts)
        
    # Ensure types are Python int and list
    # The problem states: "integers must be Python int ... sequences must be list or tuple"
    # My code produces lists of ints.
    
    return final_result

# EVOLVE-BLOCK-END
