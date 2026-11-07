"""Rewritten solver: greedy + local search + multi-start DFS for max valid subset-sum set."""
import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin under 90s

def _is_valid_subset_sum_set(nums):
    """Check if nums satisfies the divisibility-incomparable subset sum property.
    Also checks that all subset sums are distinct (no two subsets have same sum)."""
    n = len(nums)
    # Generate all nonempty subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += nums[i]
        if s in sums:
            return False
        sums.add(s)
    # Check no two sums divide each other
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True


def _is_valid_fast(nums):
    """Faster validity check using incremental approach."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        # Too many subsets to enumerate; use the structural property
        # Actually for n up to ~20, 2^n is manageable. For larger, we need smarter.
        # Let's just do the full check for n <= 20
        pass
    
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += nums[i]
        if s in sums:
            return False
        sums.add(s)
    
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
            if sj > si * 2:  # optimization: if sj > 2*si, can't divide
                # Actually sj % si == 0 means sj = k*si for k>=2
                # So we only need to check up to sj <= si * (sj//si)
                # This optimization is tricky, let's just check all
                pass
    return True


def _greedy_construct(pool, target_size, rng):
    """Greedy: add numbers that don't violate validity."""
    result = []
    sums = set()
    remaining = list(pool)
    rng.shuffle(remaining)
    
    for num in remaining:
        if len(result) >= target_size:
            break
        # Try adding num
        new_sums = set(sums)
        # New subset sums involving num
        for s in list(sums):
            new_sums.add(s + num)
        new_sums.add(num)
        
        # Check distinctness
        if len(new_sums) != len(sums) + len(sums) + 1:
            continue
        
        # Check divisibility
        valid = True
        for s1 in new_sums:
            for s2 in new_sums:
                if s1 != s2 and s1 < s2 and s2 % s1 == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            result.append(num)
            sums = new_sums
    
    return result


def _local_search(start_set, pool, max_iters, rng, deadline):
    """Local search: try swapping elements to improve validity/size."""
    current = list(start_set)
    current_sums = set()
    for mask in range(1, 1 << len(current)):
        s = 0
        for i in range(len(current)):
            if mask >> i & 1:
                s += current[i]
        current_sums.add(s)
    
    best = list(current)
    best_size = len(current)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random element from pool
        added = False
        candidates = [x for x in pool if x not in current]
        rng.shuffle(candidates)
        
        for cand in candidates[:50]:  # limit candidates
            new_set = current + [cand]
            new_sums = set(current_sums)
            for s in list(current_sums):
                new_sums.add(s + cand)
            new_sums.add(cand)
            
            if len(new_sums) != len(current_sums) * 2 + 1:
                continue
            
            valid = True
            sorted_ns = sorted(new_sums)
            for i in range(len(sorted_ns)):
                si = sorted_ns[i]
                for j in range(i + 1, len(sorted_ns)):
                    sj = sorted_ns[j]
                    if sj % si == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current = new_set
                current_sums = new_sums
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                added = True
                break
        
        if not added:
            # Try removing one element and adding two
            if len(current) >= 2:
                for i in range(len(current)):
                    for j in range(len(candidates)):
                        for k in range(j + 1, len(candidates)):
                            new_set = current[:i] + current[i+1:] + [candidates[j], candidates[k]]
                            if len(set(new_set)) != len(new_set):
                                continue
                            # Quick check
                            new_sums = set()
                            n = len(new_set)
                            for mask in range(1, 1 << n):
                                s = 0
                                for ii in range(n):
                                    if mask >> ii & 1:
                                        s += new_set[ii]
                                if s in new_sums:
                                    break
                                new_sums.add(s)
                            else:
                                sorted_ns = sorted(new_sums)
                                valid = True
                                for ii in range(len(sorted_ns)):
                                    si = sorted_ns[ii]
                                    for jj in range(ii + 1, len(sorted_ns)):
                                        sj = sorted_ns[jj]
                                        if sj % si == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid and len(new_set) > len(current):
                                    current = new_set
                                    current_sums = new_sums
                                    if len(current) > best_size:
                                        best = list(current)
                                        best_size = len(current)
                                    break
                        if len(current) > len(best):
                            break
                    if len(current) > len(best):
                        break
    
    return best


def _dfs_construct(pool, target_size, deadline):
    """DFS with backtracking to find a valid set of target_size."""
    pool = sorted(pool, reverse=True)
    result = []
    
    def dfs(start_idx, current, current_sums):
        if time.time() > deadline:
            return False
        if len(current) == target_size:
            result.extend(current)
            return True
        
        for i in range(start_idx, len(pool)):
            num = pool[i]
            # Try adding num
            new_sums = set(current_sums)
            for s in list(current_sums):
                new_sums.add(s + num)
            new_sums.add(num)
            
            if len(new_sums) != len(current_sums) + len(current_sums) + 1:
                continue
            
            # Check divisibility
            valid = True
            sorted_ns = sorted(new_sums)
            for a in range(len(sorted_ns)):
                sa = sorted_ns[a]
                for b in range(a + 1, len(sorted_ns)):
                    sb = sorted_ns[b]
                    if sb % sa == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                if dfs(i + 1, current + [num], new_sums):
                    return True
        
        return False
    
    # Start with empty set
    if dfs(0, [], set()):
        return result
    return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = random.Random(42)
    
    # Full pool
    full_pool = list(range(1, 257))
    
    best_result = [31, 32, 34]  # seed
    best_size = 3
    
    # Strategy 1: Multi-start greedy with different orderings
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        
        # Use a biased pool: prefer numbers that are "less divisible"
        # Higher numbers tend to have fewer divisors among subset sums
        pool = list(range(1, 257))
        rng.shuffle(pool)
        
        # Bias toward larger numbers
        pool.sort(key=lambda x: (-x, rng.random()))
        
        g_result = _greedy_construct(pool, 30, rng)
        if len(g_result) > best_size:
            best_size = len(g_result)
            best_result = g_result
    
    # Strategy 2: Local search from greedy results
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        pool = list(range(1, 257))
        rng.shuffle(pool)
        pool.sort(key=lambda x: (-x, rng.random()))
        
        g_result = _greedy_construct(pool, 30, rng)
        if len(g_result) < best_size:
            g_result = best_result[:]
        
        ls_result = _local_search(g_result, pool, 500, rng, deadline - 10)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_result = ls_result
    
    # Strategy 3: Targeted DFS for specific sizes
    # Try to build up incrementally
    current_best = sorted(best_result)
    
    # Try to extend current best
    for target in range(best_size + 1, min(best_size + 5, 31)):
        if time.time() > deadline - 15:
            break
        
        # Use DFS to try to find a set of size target
        pool = [x for x in range(1, 257) if x not in current_best]
        # Add some flexibility: allow replacing some elements
        dfs_result = _dfs_construct(pool, target - len(current_best), deadline - 15)
        if dfs_result:
            candidate = current_best + dfs_result
            if len(set(candidate)) == len(candidate):
                # Verify
                if _is_valid_fast(candidate):
                    best_result = candidate
                    best_size = len(candidate)
                    current_best = sorted(candidate)
    
    # Strategy 4: Random restarts with mixed strategy
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Create a random subset of the pool
        pool = rng.sample(range(1, 257), min(100, len(range(1, 257))))
        pool.sort(reverse=True)
        
        # Greedy from this pool
        g_result = _greedy_construct(pool, 30, rng)
        if len(g_result) > best_size:
            best_size = len(g_result)
            best_result = g_result
    
    # Final verification
    if _is_valid_fast(best_result):
        return [int(x) for x in best_result]
    
    # Fallback: return seed
    return [31, 32, 34]
