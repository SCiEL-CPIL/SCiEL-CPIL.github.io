import random
import time

P = {'n': 40, 'maximum_size': 18}

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_SHIFTS = [0, 7, 13, 19, 24, 29, 34, 39, 44, 49, 54, 59]

def _pack_factor(n):
    val = 0
    for i, p in enumerate(_PRIMES):
        count = 0
        while n % p == 0:
            count += 1
            n //= p
        val |= (count << _SHIFTS[i])
    return val

_PACKED = {i: _pack_factor(i) for i in range(2, 41)}

def _product_distinct(nums):
    k = len(nums)
    if k == 0:
        return True
    
    factors = [_PACKED[x] for x in nums]
    
    num_subsets = 1 << k
    seen = set()
    seen.add(0)
    
    # Use list for DP to avoid recomputation
    # For k=18, 262144 entries.
    # Python list of ints is fast.
    dp = [0] * num_subsets
    dp[0] = 0
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        val = dp[prev_mask] + factors[idx]
        dp[mask] = val
        
        if val in seen:
            return False
        seen.add(val)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_size = 0
    
    # Deterministic Core: Prime Powers
    # Base 2: 2, 4, 16 (1, 2, 4)
    # Base 3: 3, 9 (1, 2)
    # Base 5: 5, 25 (1, 2)
    # Others: 7, 11, 13, 17, 19, 23, 29, 31, 37
    core = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if _product_distinct(core):
        best_set = core[:]
        best_size = len(core)
    else:
        # Fallback to greedy if core fails (shouldn't happen)
        best_set = [2]
        best_size = 1

    # Local Search to expand from core
    def _local_search(nums, time_limit=10.0):
        current = list(nums)
        if not _product_distinct(current):
            return None
        
        best = current
        best_size = len(current)
        start_time_ls = time.time()
        
        while time.time() - start_time_ls < time_limit and best_size < P['maximum_size']:
            improved = False
            
            # Try adding
            for candidate in range(2, P['n'] + 1):
                if candidate not in best:
                    new_set = best + [candidate]
                    if _product_distinct(new_set):
                        best = new_set
                        best_size = len(best)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try single swap
            for i in range(len(best)):
                for candidate in range(2, P['n'] + 1):
                    if candidate == best[i] or candidate in best:
                        continue
                    new_set = [x for j, x in enumerate(best) if j != i]
                    new_set.append(candidate)
                    if _product_distinct(new_set):
                        best = new_set
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return best

    # Run local search from core
    if time.time() < deadline - 5.0:
        result = _local_search(best_set, time_limit=deadline - time.time() - 2.0)
        if result and len(result) > best_size:
            best_size = len(result)
            best_set = result[:]

    # If still not 18, try random restarts
    if best_size < P['maximum_size']:
        attempts = 0
        while time.time() < deadline - 2.0 and attempts < 50:
            attempts += 1
            # Random start near core size
            size = random.randint(10, 15)
            candidate = random.sample(range(2, 41), size)
            if not _product_distinct(candidate):
                continue
            
            improved = _local_search(candidate, time_limit=1.0)
            if improved and len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
            
            if best_size >= P['maximum_size']:
                break

    if best_set is None:
        best_set = [2]
    
    return [int(x) for x in best_set]
