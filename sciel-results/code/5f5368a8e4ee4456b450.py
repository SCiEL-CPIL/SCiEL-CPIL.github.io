import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s
    
    N = math.factorial(12)
    
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
    
    # We want to find the smallest M such that N can be decomposed 
    # into distinct factors from valid_divs that are <= M.
    
    # Precompute a set of valid divisors for fast lookup? 
    # Not strictly needed for the DFS logic, but good for checking "is x a divisor".
    # All factors must be divisors of N.
    
    def find_decomp_for_M(M):
        # Returns a list of factors if found, else None.
        # Factors must be distinct, > 12, <= M, product N.
        allowed = [d for d in valid_divs if d <= M]
        
        if not allowed:
            return None
        
        # Precompute suffix products
        # suffix_prods[i] = product of allowed[i:]
        # If suffix_prods[i] < remaining, we can't reach 'remaining'.
        suffix_prods = [1] * (len(allowed) + 1)
        for i in range(len(allowed) - 1, -1, -1):
            suffix_prods[i] = suffix_prods[i+1] * allowed[i]
            
        # If total product of all allowed < N, impossible
        if suffix_prods[0] < N:
            return None

        def backtrack(remaining, idx):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= len(allowed):
                return None
            
            # Prune: if the smallest available factor is larger than remaining,
            # we can't pick any more factors (since all are >= allowed[idx]).
            # And we still need to reduce remaining to 1 (which requires multiplying by 1, i.e., picking no more, 
            # but remaining != 1, so we fail).
            if allowed[idx] > remaining:
                return None
                
            # Prune: if product of all remaining allowed factors < remaining
            if suffix_prods[idx] < remaining:
                return None
                
            # Option 1: Take allowed[idx]
            # If we take it, remaining becomes remaining // allowed[idx]
            # Only valid if remaining is divisible by allowed[idx]
            if remaining % allowed[idx] == 0:
                res = backtrack(remaining // allowed[idx], idx + 1)
                if res is not None:
                    return [allowed[idx]] + res
            
            # Option 2: Skip allowed[idx]
            res = backtrack(remaining, idx + 1)
            if res is not None:
                return res
            
            return None
        
        res = backtrack(N, 0)
        if res:
            return res
        return None

    # Iterate M over valid_divs
    for M in valid_divs:
        if time.time() > deadline:
            break
            
        res = find_decomp_for_M(M)
        if res is not None:
            return res
    
    # Fallback: [N]
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
