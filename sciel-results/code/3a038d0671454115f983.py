# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns 128 lists representing targets 1..128.
    Each list contains distinct 5-smooth numbers summing to the target.
    The set of numbers in each list forms a divisibility antichain.
    Minimizes the total number of summands across all targets.
    """
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit

    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_numbers.sort()
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility using bitsets for fast compatibility checks
    # conflict_mask[i] is a bitset where bit j is set if smooth_numbers[i] divides or is divided by smooth_numbers[j]
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        mask = 0
        for j in range(n_smooth):
            if i != j:
                vj = smooth_numbers[j]
                if vj % vi == 0 or vi % vj == 0:
                    mask |= (1 << j)
        conflict_mask[i] = mask
    
    # best_for_sum[s] = (min_size, tuple_of_indices)
    best_for_sum = [None] * 129
    
    # Phase 1: Quick per-target DFS to find valid solutions
    def dfs_for_target(target, start, rem, path, path_mask):
        if time.time() - start_time > timeout:
            return None
        if rem == 0:
            return list(path)
        if rem < 0:
            return None
        # Try larger values first to minimize count
        for i in range(n_smooth - 1, start - 1, -1):
            if i < start:
                break
            val = smooth_numbers[i]
            if val > rem:
                continue
            # Check compatibility using bitset
            if path_mask & conflict_mask[i]:
                continue
            path.append(i)
            new_mask = path_mask | (1 << i)
            res = dfs_for_target(target, i + 1, rem - val, path, new_mask)
            if res is not None:
                return res
            path.pop()
        return None
    
    # Phase 2: Global backtracking to find minimal antichains for all sums simultaneously
    def backtrack(start_idx, current_sum, current_size, current_set, current_mask):
        if time.time() - start_time > timeout:
            return False
            
        if current_sum > 0 and current_sum <= 128:
            if best_for_sum[current_sum] is None or current_size < best_for_sum[current_sum][0]:
                best_for_sum[current_sum] = (current_size, tuple(current_set))
        
        if current_sum > 128:
            return True
            
        for i in range(start_idx, n_smooth):
            val = smooth_numbers[i]
            if val > 128 - current_sum:
                break
            
            # Check compatibility using bitset
            if current_mask & conflict_mask[i]:
                continue
            
            current_set.append(i)
            new_mask = current_mask | (1 << i)
            if not backtrack(i + 1, current_sum + val, current_size + 1, current_set, new_mask):
                current_set.pop()
                return False
            current_set.pop()
        
        return True

    # Run global backtracking with time limit
    if time.time() - start_time < timeout * 0.7:
        backtrack(0, 0, 0, [], 0)
    
    # Fill in any missing targets
    final_answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            size, indices = best_for_sum[n]
            nums = [smooth_numbers[i] for i in indices]
            final_answers.append(nums)
        else:
            # Fallback: per-target DFS
            ans = dfs_for_target(n, 0, n, [], 0)
            if ans:
                final_answers.append([smooth_numbers[i] for i in ans])
            else:
                # Last resort: find any valid representation
                ans2 = _find_any_representation(n, smooth_numbers, conflict_mask, n_smooth)
                if ans2:
                    final_answers.append(ans2)
                else:
                    final_answers.append([n] if n in smooth_numbers else [1, n-1])
                
    return final_answers

def _find_any_representation(target, smooth_numbers, conflict_mask, n_smooth):
    """Find any valid antichain representation for target."""
    def dfs(start, rem, path, path_mask):
        if rem == 0:
            return list(path)
        if rem < 0:
            return None
        for i in range(start, n_smooth):
            val = smooth_numbers[i]
            if val > rem:
                break
            if path_mask & conflict_mask[i]:
                continue
            path.append(i)
            new_mask = path_mask | (1 << i)
            res = dfs(i + 1, rem - val, path, new_mask)
            if res is not None:
                return res
            path.pop()
        return None
    
    indices = dfs(0, target, [], 0)
    if indices:
        return [smooth_numbers[i] for i in indices]
    return None

# EVOLVE-BLOCK-END
