"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

N = 5040
TARGET = 4979
DEADLINE_S = 55  # safety margin before 60s kill

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_DIVS = _divisors_of(N)
_DIVS_SET = set(_DIVS)
_DIVS_DESC = sorted(_DIVS, reverse=True)

def _greedy_solution():
    """Greedy: pick largest divisor <= remainder."""
    rem = TARGET
    chosen = []
    for d in _DIVS_DESC:
        if d <= rem:
            chosen.append(d)
            rem -= d
            if rem == 0:
                break
    if rem == 0:
        return chosen
    return None

def _dp_solution():
    """DP to find minimum number of distinct divisors summing to TARGET.
    Uses bitmask to track which divisors are used."""
    # Filter divisors <= TARGET (larger ones can't be used)
    divs = [d for d in _DIVS if d <= TARGET]
    num_divs = len(divs)
    
    INF = float('inf')
    min_count = [INF] * (TARGET + 1)
    mask = [0] * (TARGET + 1)
    min_count[0] = 0
    mask[0] = 0
    
    start = time.time()
    for idx, d in enumerate(divs):
        if time.time() - start > DEADLINE_S - 5:
            break
        # Update dp in reverse order to avoid using same divisor twice
        for s in range(TARGET, d - 1, -1):
            prev_count = min_count[s - d]
            if prev_count != INF:
                new_count = prev_count + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    mask[s] = mask[s - d] | (1 << idx)
    
    if min_count[TARGET] != INF:
        result_mask = mask[TARGET]
        result = []
        for idx in range(num_divs):
            if result_mask & (1 << idx):
                result.append(divs[idx])
        return result
    return None

def _small_k_search(k):
    """Search for exactly k distinct divisors summing to TARGET.
    Returns the list if found, else None."""
    divs = _DIVS_DESC
    n = len(divs)
    start = time.time()
    
    if k == 1:
        if TARGET in _DIVS_SET:
            return [TARGET]
        return None
    
    if k == 2:
        for d1 in divs:
            if d1 >= TARGET:
                break
            d2 = TARGET - d1
            if d2 in _DIVS_SET and d2 > d1:
                return [d1, d2]
        return None
    
    if k == 3:
        for i, d1 in enumerate(divs):
            if d1 >= TARGET:
                break
            rem1 = TARGET - d1
            for j in range(i + 1, n):
                d2 = divs[j]
                if d2 >= rem1:
                    break
                d3 = rem1 - d2
                if d3 in _DIVS_SET and d3 > d2:
                    return [d1, d2, d3]
        return None
    
    if k == 4:
        for i, d1 in enumerate(divs):
            if d1 >= TARGET:
                break
            rem1 = TARGET - d1
            for j in range(i + 1, n):
                d2 = divs[j]
                if d2 >= rem1:
                    break
                rem2 = rem1 - d2
                for l in range(j + 1, n):
                    d3 = divs[l]
                    if d3 >= rem2:
                        break
                    d4 = rem2 - d3
                    if d4 in _DIVS_SET and d4 > d3:
                        return [d1, d2, d3, d4]
        return None
    
    if k == 5:
        for i, d1 in enumerate(divs):
            if d1 >= TARGET:
                break
            rem1 = TARGET - d1
            for j in range(i + 1, n):
                d2 = divs[j]
                if d2 >= rem1:
                    break
                rem2 = rem1 - d2
                for l in range(j + 1, n):
                    d3 = divs[l]
                    if d3 >= rem2:
                        break
                    rem3 = rem2 - d3
                    for m in range(l + 1, n):
                        d4 = divs[m]
                        if d4 >= rem3:
                            break
                        d5 = rem3 - d4
                        if d5 in _DIVS_SET and d5 > d4:
                            return [d1, d2, d3, d4, d5]
        return None
    
    # For k >= 6, use combinations with pruning
    # But this is expensive, so we'll use a different approach
    # Actually, let's just return None and let DP handle it
    return None

def construct():
    start = time.time()
    
    # Step 1: Try small k values first (1 to 5) for fast exact search
    for k in range(1, 6):
        if time.time() - start > DEADLINE_S - 10:
            break
        sol = _small_k_search(k)
        if sol is not None:
            # Validate
            if len(set(sol)) == len(sol) and sum(sol) == TARGET and all(d in _DIVS_SET for d in sol):
                return sol
    
    # Step 2: Use DP to find minimum number of divisors
    sol = _dp_solution()
    if sol is not None:
        # Validate
        if len(set(sol)) == len(sol) and sum(sol) == TARGET and all(d in _DIVS_SET for d in sol):
            return sol
    
    # Step 3: Fallback to greedy
    sol = _greedy_solution()
    if sol is not None:
        return sol
    
    # Final fallback: return empty (should not happen)
    return []

# EVOLVE-BLOCK-END
