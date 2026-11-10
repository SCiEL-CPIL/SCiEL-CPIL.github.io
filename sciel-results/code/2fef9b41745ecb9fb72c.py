import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses bitmask-accelerated local search with multiple random starts.
    """
    m = 205
    deadline = time.time() + 580.0  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitmasks
    # adj_mask[i] is a bitmask where bit j is 1 if (i-j) is in forbidden
    # Actually, it's easier to define conflict_mask[i] as bitmask of nodes conflicting with i
    # If i and j conflict, then j is in conflict_mask[i] and i is in conflict_mask[j]
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_valid_set(selected_mask):
        # Check if any two elements in selected_mask conflict
        # For each i in selected, check if (selected & conflict_mask[i]) has any bits other than i
        # But since conflict_mask[i] doesn't include i, we just check if selected & conflict_mask[i] is non-zero
        # However, we need to ensure we don't count self. conflict_mask[i] does not have bit i set.
        # So if (selected & conflict_mask[i]) != 0, it means there is a conflict.
        # We must be careful: if we iterate through set bits, we might double count.
        # A valid set S is independent if for all i in S, (S & conflict_mask[i]) == 0 is NOT quite right because 
        # if j in S and j conflicts with i, then j is in conflict_mask[i].
        # So we check: for all i in S, (S & conflict_mask[i]) must be 0? 
        # No, if i and j conflict, j is in conflict_mask[i]. So S & conflict_mask[i] will contain j.
        # Thus, for an independent set, for every i in S, (S & conflict_mask[i]) must be 0? 
        # Wait, if S={i, j} and they conflict, then j is in conflict_mask[i]. 
        # So S & conflict_mask[i] has bit j set. 
        # So yes, if S is independent, for all i in S, (S & conflict_mask[i]) == 0.
        
        # Optimization: We can just check if any pair conflicts.
        # But checking all pairs is O(N^2).
        # The bitmask check is O(N * (bit operations)).
        # Let's stick to the pairwise check for validity verification at the end.
        # During search, we maintain validity.
        pass

    def get_conflicts(selected_mask, candidate):
        """
        Returns True if candidate conflicts with any node in selected_mask.
        """
        # candidate is an integer index 0..204
        # conflict_mask[candidate] has bits set for all nodes conflicting with candidate.
        # If any of those bits are also set in selected_mask, there is a conflict.
        return (selected_mask & conflict_mask[candidate]) != 0

    def greedy_random(seed):
        """
        Generate a random greedy solution.
        """
        random.seed(seed)
        selected_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for c in candidates:
            if not get_conflicts(selected_mask, c):
                selected_mask |= (1 << c)
                
        return selected_mask

    def local_search(start_mask):
        """
        Improve a solution using 1-add, 2-swap (remove 1, add 2), and 1-swap (remove 1, add 1) moves.
        Uses a simple hill-climbing with random restarts/shakes if stuck.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        no_improvement_count = 0
        max_iterations = 20000 # Increased iterations for more thorough search
        shake_count = 0
        
        while no_improvement_count < 100 and max_iterations > 0 and time.time() < deadline:
            max_iterations -= 1
            improved = False
            
            # 1. Try 1-Add: Add a node that doesn't conflict
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if not get_conflicts(current_mask, c):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
            
            if improved:
                no_improvement_count = 0
                continue
            
            # 2. Try 2-Swap: Remove one node i, add two nodes j, k
            # This increases size by 1.
            # Check if there exists i in S, j, k not in S such that 
            # {S \ {i}} U {j, k} is independent.
            # This means j, k don't conflict with S\{i}, and j doesn't conflict with k.
            
            current_indices = [i for i in range(m) if current_mask & (1 << i)]
            
            found_2swap = False
            for i in current_indices:
                mask_without_i = current_mask & ~(1 << i)
                
                # Candidates are nodes not in current_mask
                # We need to find two candidates j, k that are compatible with mask_without_i and each other.
                # To speed this up, we can iterate j, then find k.
                
                # Optimization: Precompute list of candidates not in current_mask
                # But current_mask changes. Let's just iterate.
                
                # For each j not in current_mask:
                #   If j is compatible with mask_without_i:
                #      For each k not in current_mask, k != j:
                #         If k is compatible with mask_without_i and k compatible with j:
                #            Perform swap.
                
                # This is O(N^3) worst case, but we break early.
                # N=205. 205^3 is ~8M. Might be too slow if done every iteration.
                # We only do this if 1-add fails.
                
                # Let's limit the search for 2-swap to avoid hanging.
                # We can iterate j, and for each j, check if there is ANY k.
                
                # A faster check:
                # For a fixed i, let ValidCandidates be {c not in S | c compatible with S\{i}}.
                # We need to find an edge (j,k) in the graph of ValidCandidates where j,k are not connected by conflict?
                # No, we need j,k to be compatible.
                
                # Let's just do the nested loops but with a small limit on j iterations or break early.
                
                # Actually, a simpler heuristic:
                # Try to find a pair (j, k) that are both compatible with mask_without_i.
                
                # Let's iterate j.
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    # Check if j is compatible with mask_without_i
                    if mask_without_i & conflict_mask[j]:
                        continue
                    
                    # j is a candidate. Now find k.
                    # k must not be in current_mask, k != j, k compatible with mask_without_i, k compatible with j.
                    for k in range(j + 1, m):
                        if current_mask & (1 << k):
                            continue
                        # Check k compatible with mask_without_i
                        if mask_without_i & conflict_mask[k]:
                            continue
                        # Check k compatible with j
                        if (1 << j) & conflict_mask[k]:
                            continue
                        
                        # Found valid 2-swap
                        new_mask = mask_without_i | (1 << j) | (1 << k)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        found_2swap = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if found_2swap:
                no_improvement_count = 0
                continue

            # 3. Try 1-Swap: Remove one node i, add one node j (size stays same)
            # This helps escape local optima.
            # We use a random element for i to ensure diversity, or iterate.
            # Let's pick a random i from current_mask.
            if current_indices:
                i = random.choice(current_indices)
                mask_without_i = current_mask & ~(1 << i)
                
                # Find a random j not in current_mask that is compatible with mask_without_i
                # To be more effective, pick j that has high degree to blocked nodes?
                # For now, just pick any valid j.
                
                # Collect valid j's
                valid_js = []
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if not (mask_without_i & conflict_mask[j]):
                        valid_js.append(j)
                
                if valid_js:
                    j = random.choice(valid_js)
                    current_mask = mask_without_i | (1 << j)
                    # Size is same, so no best_size update unless we want to track structure
                    # We consider this an "improvement" in terms of exploration, so reset counter
                    # but don't update best_size.
                    no_improvement_count = 0
                    continue
            
            # If no moves found, we are stuck.
            # Apply a "shake": randomly remove a few nodes to allow re-adding.
            # Or just increment no_improvement_count.
            no_improvement_count += 1
            
            # If stuck for a long time, perform a random restart from a perturbed version
            if no_improvement_count > 50:
                # Perturb: remove 1-3 random nodes
                num_remove = random.randint(1, 3)
                if current_size > num_remove:
                    indices_to_remove = random.sample([i for i in range(m) if current_mask & (1 << i)], num_remove)
                    for idx in indices_to_remove:
                        current_mask &= ~(1 << idx)
                    current_size = bin(current_mask).count('1')
                    no_improvement_count = 0 # Reset to allow more search
                    shake_count += 1
                    if shake_count > 10:
                        break # Give up on this start
                
        return best_mask, best_size

    # Main Search Loop
    best_global_mask = 0
    best_global_size = 0
    
    # Start with a few deterministic/random seeds
    num_starts = 50
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Use different seeds
        seed = i * 12345 + 67890
        init_mask = greedy_random(seed)
        
        # Local search
        improved_mask, improved_size = local_search(init_mask)
        
        if improved_size > best_global_size:
            best_global_size = improved_size
            best_global_mask = improved_mask
            
        # If we found a very large set, we can stop early? 
        # Max possible size is roughly m / (avg_degree+1). 
        # Squares mod 205: 205 = 5 * 41.
        # Number of squares is roughly m/2. Degree is roughly 100.
        # Max independent set is roughly 205 / 101 ~ 2. That's too low.
        # Wait, the graph is not regular in a simple way.
        # Let's check the size of forbidden set.
        # Squares mod 5: {1, 4} (2 non-zero)
        # Squares mod 41: 20 non-zero.
        # By CRT, number of squares mod 205 is 2 * 20 = 40? 
        # No. x^2 mod 5 in {1,4}. x^2 mod 41 in 20 residues.
        # Total non-zero squares = 40.
        # So degree is 40.
        # Max independent set size >= 205 / (40+1) ~ 5.
        # Actually, the seed score 1.0 corresponds to some size.
        # The knowledge base said 1.333 was achieved.
        # If seed is 1.0, and 1.333 is better, the sizes are likely 9 and 12? 
        # Or 10 and 13?
        # Let's assume we are looking for size > 10.

    # Convert best mask to array
    result = []
    for i in range(m):
        if best_global_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
