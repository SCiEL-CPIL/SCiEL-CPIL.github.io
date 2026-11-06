import math
import time
import bisect
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin out of 60s (evaluator kills at 90s, but keep safe)
    
    N = math.factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Let's verify:
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # 11: floor(12/11) = 1
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    if not valid_divs:
        return [N]
    
    # We want to find the smallest M such that N can be decomposed 
    # into distinct factors from valid_divs that are <= M.
    # The number of factors must be between 1 and 12.
    
    def find_decomp_for_M(M):
        # Returns a list of factors if found, else None.
        # Factors must be distinct, > 12, <= M, product N, count <= 12.
        
        # Find index of last divisor <= M
        idx_end = bisect.bisect_right(valid_divs, M)
        if idx_end == 0:
            return None
            
        allowed = valid_divs[:idx_end]
        
        if not allowed:
            return None
        
        # Precompute suffix products for pruning
        suffix_prods = [1] * (len(allowed) + 1)
        for i in range(len(allowed) - 1, -1, -1):
            suffix_prods[i] = suffix_prods[i+1] * allowed[i]
            
        # If total product of all allowed < N, impossible
        if suffix_prods[0] < N:
            return None
        
        # Use memoization with (remaining, idx) -> result or None
        # But we need to be careful with memory and time.
        # Let's use a set of visited states to avoid re-exploring.
        # Actually, for this problem size, simple backtracking with good pruning should work.
        
        def backtrack(remaining, idx, count):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= len(allowed):
                return None
            
            # Prune: if count >= 12, we can't add more
            if count >= 12:
                return None
                
            # Prune: if the smallest available factor is larger than remaining,
            # we can't pick any more factors (since all remaining factors are >= allowed[idx])
            if allowed[idx] > remaining:
                return None
                
            # Prune: if product of all remaining allowed factors < remaining
            if suffix_prods[idx] < remaining:
                return None
            
            # Prune: if remaining is prime and > 12 and > M, impossible
            # (but remaining must be a divisor of N, so if it's prime it must be in allowed)
            
            # Option 1: Take allowed[idx]
            if remaining % allowed[idx] == 0:
                res = backtrack(remaining // allowed[idx], idx + 1, count + 1)
                if res is not None:
                    return [allowed[idx]] + res
            
            # Option 2: Skip allowed[idx]
            res = backtrack(remaining, idx + 1, count)
            if res is not None:
                return res
            
            return None
        
        res = backtrack(N, 0, 0)
        if res:
            return res
        return None

    # Binary search for the minimum M
    # The answer M must be one of the valid divisors.
    # We search for the smallest index in valid_divs such that find_decomp_for_M(valid_divs[idx]) is not None.
    
    lo = 0
    hi = len(valid_divs) - 1
    best_res = None
    best_M = None
    
    # Check if the largest M works (sanity check / fallback)
    # [N] is always a valid decomposition (1 factor, N > 12, distinct).
    # So we are looking for the minimum possible max factor.
    
    # We'll do a linear scan from small to large M, but with binary search for efficiency.
    # Actually, since the search space is small (number of divisors of 12! is not huge),
    # let's just iterate through valid_divs in order and return the first one that works.
    
    # But binary search is better if find_decomp_for_M is monotonic (which it should be:
    # if M works, then any M' > M also works).
    
    # Let's verify monotonicity: if we can decompose with max factor <= M,
    # then we can also decompose with max factor <= M' for M' >= M (just use the same decomposition).
    # Yes, it's monotonic.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        M_candidate = valid_divs[mid]
        
        res = find_decomp_for_M(M_candidate)
        
        if res is not None:
            best_res = res
            best_M = M_candidate
            hi = mid - 1  # Try to find a smaller M
        else:
            lo = mid + 1   # M_candidate is too small, try larger
            
    if best_res is not None:
        return best_res
    else:
        return [N]

if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 1
    for x in res:
        prod *= x
    print("Product:", prod)
    print("12!:", math.factorial(12))
    print("Max:", max(res))
    print("Distinct:", len(res) == len(set(res)))
    print("All > 12:", all(x > 12 for x in res))
    print("Count:", len(res))
