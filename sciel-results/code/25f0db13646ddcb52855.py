import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute comparability? 
    # For each mask i, we can store a bitset of masks j such that i subset j or j subset i.
    # However, 4096 * 4096 bits is 2MB. 
    # Let's just compute on the fly for the 40 active sets, it's fast enough.
    
    def get_initial_solution():
        # Take the 40 smallest subsets of the first 6 vertices
        # Vertices 0..5
        core_size = 6
        subsets = []
        for mask in range(1 << core_size):
            subsets.append(mask)
        
        # Sort by popcount, then by value
        subsets.sort(key=lambda x: (bin(x).count('1'), x))
        
        # We need 40 distinct subsets. 
        # The first 40 in this sorted list are the smallest.
        return subsets[:m]
    
    def count_comparable(sets):
        cnt = 0
        for i in range(len(sets)):
            si = sets[i]
            for j in range(i + 1, len(sets)):
                sj = sets[j]
                if (si & sj) == si:
                    cnt += 1
                elif (si & sj) == sj:
                    cnt += 1
        return cnt
    
    # Initial solution
    current_sets = get_initial_solution()
    current_score = count_comparable(current_sets)
    
    # Track which masks are in the solution for O(1) lookup
    in_solution = [False] * total_masks
    for s in current_sets:
        in_solution[s] = True
    
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Local Search
    # We iterate through indices and try to replace current_sets[i] with a new mask
    # To avoid O(m * 4096) full scans which might be slow in pure Python if not optimized,
    # we can be smart. But 40 * 4096 = 163,840 checks per full pass.
    # Each check involves comparing with 39 other sets. 163,840 * 39 ~ 6.4M ops.
    # This is very fast in Python (likely < 0.5s per full pass).
    # So we can do many passes.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Shuffle indices to avoid bias? Or just iterate.
        # Iterating is deterministic and fine.
        for i in range(m):
            if (time.time() - start_time) > deadline:
                break
                
            old_mask = current_sets[i]
            best_gain = 0
            best_new_mask = -1
            
            # Calculate current contribution of old_mask
            # Actually, we can just calculate the gain for each candidate.
            # Gain = (comparables of new_mask with others) - (comparables of old_mask with others)
            
            # Pre-compute comparables of old_mask with others (excluding i)
            old_comp_count = 0
            for j in range(m):
                if i == j: continue
                sj = current_sets[j]
                if (old_mask & sj) == old_mask or (old_mask & sj) == sj:
                    old_comp_count += 1
            
            # Try all possible new masks
            # Optimization: Only consider masks not in solution
            for new_mask in range(total_masks):
                if in_solution[new_mask]:
                    continue
                if new_mask == old_mask:
                    continue
                
                new_comp_count = 0
                for j in range(m):
                    if i == j: continue
                    sj = current_sets[j]
                    if (new_mask & sj) == new_mask or (new_mask & sj) == sj:
                        new_comp_count += 1
                
                gain = new_comp_count - old_comp_count
                if gain > best_gain:
                    best_gain = gain
                    best_new_mask = new_mask
            
            if best_new_mask != -1:
                # Apply swap
                in_solution[old_mask] = False
                in_solution[best_new_mask] = True
                current_sets[i] = best_new_mask
                current_score += best_gain
                improved = True
                # Restart loop to re-evaluate with new configuration
                # Breaking the inner loop and restarting outer is standard for steepest ascent
                # But we can just continue to next i, or break to restart.
                # Restarting is safer for convergence.
                break
                
    # Convert masks back to lists of integers
    result = []
    for mask in current_sets:
        s = []
        for k in range(n):
            if mask & (1 << k):
                s.append(k)
        result.append(s)
        
    return result
