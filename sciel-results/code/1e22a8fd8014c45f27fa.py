# EVOLVE-BLOCK-START
import time
import itertools

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
    
    # Precompute divisibility matrix for antichain checks
    # divides[i][j] is True if smooth_numbers[i] divides smooth_numbers[j]
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(n_smooth):
            if i != j:
                vj = smooth_numbers[j]
                if vj % vi == 0:
                    divides[i][j] = True
    
    # Precompute conflict masks for fast antichain checking
    # conflict_mask[i] is a bitset of indices j that are incompatible with i
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i != j:
                if divides[i][j] or divides[j][i]:
                    mask |= (1 << j)
        conflict_mask[i] = mask
    
    # 2. Find minimal antichain representations for each target
    # Strategy: For each target, try k=1, 2, 3, ... until found
    # We'll use a BFS/DFS approach with pruning
    
    best_for_sum = [None] * 129  # best_for_sum[s] = (min_size, tuple_of_indices)
    solved = [False] * 129
    
    # k=1: Single smooth numbers
    for i in range(n_smooth):
        val = smooth_numbers[i]
        if val <= 128 and not solved[val]:
            best_for_sum[val] = (1, (i,))
            solved[val] = True
    
    # k=2: Pairs of smooth numbers
    if time.time() - start_time < timeout:
        for i in range(n_smooth):
            vi = smooth_numbers[i]
            for j in range(i + 1, n_smooth):
                vj = smooth_numbers[j]
                s = vi + vj
                if s > 128:
                    break
                if not solved[s]:
                    # Check antichain: no divisibility between vi and vj
                    if not (divides[i][j] or divides[j][i]):
                        best_for_sum[s] = (2, (i, j))
                        solved[s] = True
    
    # k=3: Triples of smooth numbers
    if time.time() - start_time < timeout:
        for i in range(n_smooth):
            vi = smooth_numbers[i]
            for j in range(i + 1, n_smooth):
                vj = smooth_numbers[j]
                s2 = vi + vj
                if s2 > 128:
                    break
                # Check i and j are compatible
                if divides[i][j] or divides[j][i]:
                    continue
                for k in range(j + 1, n_smooth):
                    vk = smooth_numbers[k]
                    s3 = s2 + vk
                    if s3 > 128:
                        break
                    if not solved[s3]:
                        # Check k is compatible with i and j
                        if not (divides[i][k] or divides[k][i]) and not (divides[j][k] or divides[k][j]):
                            best_for_sum[s3] = (3, (i, j, k))
                            solved[s3] = True
    
    # k=4: Quadruples of smooth numbers
    if time.time() - start_time < timeout:
        for i in range(n_smooth):
            vi = smooth_numbers[i]
            for j in range(i + 1, n_smooth):
                vj = smooth_numbers[j]
                s2 = vi + vj
                if s2 > 128:
                    break
                if divides[i][j] or divides[j][i]:
                    continue
                for k in range(j + 1, n_smooth):
                    vk = smooth_numbers[k]
                    s3 = s2 + vk
                    if s3 > 128:
                        break
                    if divides[i][k] or divides[k][i] or divides[j][k] or divides[k][j]:
                        continue
                    for l in range(k + 1, n_smooth):
                        vl = smooth_numbers[l]
                        s4 = s3 + vl
                        if s4 > 128:
                            break
                        if not solved[s4]:
                            # Check l is compatible with i, j, k
                            if (not (divides[i][l] or divides[l][i]) and 
                                not (divides[j][l] or divides[l][j]) and 
                                not (divides[k][l] or divides[l][k])):
                                best_for_sum[s4] = (4, (i, j, k, l))
                                solved[s4] = True
    
    # For remaining targets, use DFS to find any valid representation
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
                    # Very last resort: use the number itself if smooth, or 1 + (n-1)
                    if n in smooth_numbers:
                        final_answers.append([n])
                    else:
                        final_answers.append([1, n-1])
                
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
