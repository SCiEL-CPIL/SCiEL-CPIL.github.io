import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s deadline to be safe
    
    P = math.factorial(12)
    best_max = P # Initial best is single factor P
    best_factors = [P]
    
    # Prime factors of 12! are 2, 3, 5, 7, 11
    # We don't strictly need prime factorization for DFS if we just try divisors,
    # but trying divisors is efficient.
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    # We want to find sets of k distinct integers > 12
    # k can be 1 to 7
    
    def dfs(rem, min_f, max_k, curr, current_max):
        nonlocal best_max, best_factors
        
        # Check time
        if time.time() > deadline:
            return
            
        if rem == 1:
            # We have a valid set
            # curr must contain factors > 12
            # Since min_f starts at 13 and increases, all are > 12
            if curr:
                m = max(curr)
                if m < best_max:
                    best_max = m
                    best_factors = list(curr)
            return
            
        # Pruning: if we can't fit max_k factors
        if max_k == 0:
            return
            
        # Pruning: if rem is prime and rem < min_f, impossible
        # If rem is prime, it must be one of the factors.
        # If rem < min_f, we can't pick it.
        # If rem >= min_f, we must pick it as the last factor (if max_k >= 1)
        # Wait, if rem is prime, the only divisors are 1 and rem.
        # We can't pick 1. So we must pick rem.
        # But rem must be > min_f? No, rem is the product of remaining factors.
        # If rem is prime, say p. The remaining factors must multiply to p.
        # Since factors are integers > 1, the only solution is [p].
        # So if max_k == 1, we check if rem > min_f.
        # If max_k > 1, we can't form product p with >1 integers >1.
        # So if rem is prime:
        #   if max_k == 1 and rem > min_f:
        #       if rem < best_max: update
        #   else: return
        
        # Check if rem is prime
        if rem > 1:
            is_prime = True
            for i in range(2, int(math.isqrt(rem)) + 1):
                if rem % i == 0:
                    is_prime = False
                    break
            if is_prime:
                if max_k == 1 and rem > min_f:
                    if rem < best_max:
                        best_max = rem
                        best_factors = list(curr) + [rem]
                return

        # General case: rem is composite
        # We need to pick a factor f such that:
        # 1. f > min_f
        # 2. f divides rem
        # 3. f < best_max (if we stop here or continue, f will be in the set)
        #    Actually, if we continue, subsequent factors will be > f.
        #    So if f >= best_max, we can prune.
        # 4. The remaining product rem//f must be achievable with max_k-1 factors > f.
        
        # Lower bound on the largest factor in the remaining max_k-1 factors:
        # Let R' = rem // f.
        # We need max_k-1 distinct integers > f with product R'.
        # The smallest such product is P_min = (f+1)(f+2)...(f+max_k-2).
        # If R' < P_min, impossible.
        # Also, the largest factor in the remaining set must be at least 
        # R' / (product of the smallest max_k-2 factors > f).
        # This is a bit expensive to compute exactly, but a simple bound:
        # If max_k-1 == 1, the remaining factor is R'. It must be > f and < best_max.
        # If max_k-1 > 1, the largest remaining factor L must satisfy:
        # R' = L * P_rest, where P_rest is product of max_k-2 factors > f.
        # P_rest >= (f+1)...(f+max_k-3).
        # So L <= R' / P_rest_min? No, L = R' / P_rest.
        # To minimize L, we maximize P_rest. But we want a lower bound on L to prune.
        # We want to know if it's possible to have all remaining factors < best_max.
        # The maximum product of max_k-1 distinct integers < best_max, starting from f+1, is:
        # P_max_possible = (best_max-1)(best_max-2)...(best_max-(max_k-1)).
        # Wait, they must be > f.
        # So the factors must be in [f+1, best_max-1].
        # The maximum product of max_k-1 distinct integers in this range is 
        # (best_max-1)(best_max-2)...(best_max-(max_k-1)).
        # If R' > P_max_possible, then it's impossible to form R' with max_k-1 factors all < best_max.
        
        # This bound is strong.
        
        # Iterate through possible divisors f of rem
        # f must be > min_f
        # f must be < best_max (since f is part of the set)
        # Also f <= rem // (min product of remaining max_k-1 factors)
        
        # To make it faster, iterate f from min_f+1 to some limit.
        # Limit: f must divide rem.
        # We can just iterate through divisors of rem that are > min_f.
        
        divs = get_divisors(rem)
        # Filter divs: > min_f, < best_max
        valid_divs = [d for d in divs if d > min_f and d < best_max]
        
        # Sort valid_divs to find small max factors first?
        # We want to minimize the max factor.
        # If we pick a small f, the remaining product is large, so remaining factors might be large.
        # If we pick a large f (close to best_max), remaining product is small.
        # It's better to search for solutions with small max factors.
        # The max factor of the final set is max(f, max(remaining_factors)).
        # To prune effectively, we should try to keep the current_max small.
        # But we are minimizing the GLOBAL max.
        
        # Let's just iterate through valid_divs.
        # We can sort them.
        valid_divs.sort()
        
        for f in valid_divs:
            # Check time
            if time.time() > deadline:
                return
                
            new_rem = rem // f
            new_min_f = f + 1
            new_max_k = max_k - 1
            
            # Pruning check: Can we form new_rem with new_max_k factors > new_min_f, all < best_max?
            if new_max_k == 0:
                # This implies new_rem must be 1, but we handled rem==1 at start.
                # If new_max_k is 0, we can't pick any more factors.
                # So new_rem must be 1. But rem was not 1.
                # So this case is invalid.
                continue
                
            if new_max_k == 1:
                # The only factor is new_rem.
                if new_rem > new_min_f and new_rem < best_max:
                    # Valid
                    if new_rem < best_max:
                        best_max = new_rem
                        best_factors = list(curr) + [f, new_rem]
                # If new_rem >= best_max, it won't improve, so no need to recurse (which would just return).
                # Actually, if new_rem < best_max, we update.
                # We don't need to recurse for max_k=1.
                continue
            
            # For new_max_k > 1:
            # Check if new_rem can be formed by new_max_k distinct integers in [new_min_f, best_max-1].
            # Max product of new_max_k distinct integers < best_max, starting from new_min_f:
            # The largest possible product is using the largest available numbers:
            # (best_max-1), (best_max-2), ..., (best_max-new_max_k).
            # But they must be >= new_min_f.
            # So the available range is [new_min_f, best_max-1].
            # We need to pick new_max_k distinct numbers from this range.
            # The maximum product is obtained by picking the largest new_max_k numbers:
            # best_max-1, best_max-2, ..., best_max-new_max_k.
            # But we must ensure these are >= new_min_f.
            # If best_max - new_max_k < new_min_f, then we don't have enough numbers in the range?
            # No, we just need ANY set. The max product bound is an upper bound on what's possible.
            # If new_rem > MaxPossibleProduct, it's impossible.
            
            max_prod = 1
            for i in range(new_max_k):
                factor = best_max - 1 - i
                if factor < new_min_f:
                    # Not enough numbers in the range [new_min_f, best_max-1]
                    # Actually, if factor < new_min_f, we can't use it.
                    # This means the range is too small to hold new_max_k distinct integers?
                    # No, it means the largest possible integer in the range is best_max-1.
                    # We need new_max_k distinct integers.
                    # The smallest of the largest new_max_k integers is best_max - new_max_k.
                    # If best_max - new_max_k < new_min_f, then we can't even fit the "largest" set?
                    # Wait. The set must be subset of [new_min_f, best_max-1].
                    # The size of this range is best_max - 1 - new_min_f + 1 = best_max - new_min_f.
                    # We need to pick new_max_k distinct integers.
                    # So we need best_max - new_min_f >= new_max_k.
                    if best_max - new_min_f < new_max_k:
                        max_prod = 0
                        break
                max_prod *= factor
            
            if max_prod < new_rem:
                continue
                
            # Also, check if new_rem can be formed at all?
            # Min product of new_max_k distinct integers > new_min_f:
            min_prod = 1
            for i in range(new_max_k):
                min_prod *= (new_min_f + i)
            if min_prod > new_rem:
                continue
                
            # Recurse
            curr.append(f)
            dfs(new_rem, new_min_f, new_max_k, curr, f)
            curr.pop()

    # Start search for different k
    # We can start with k=1, then k=2, etc.
    # Or just let DFS handle it by allowing up to 7 factors.
    # If we allow up to 7, it will find the best solution for any k <= 7.
    
    # Initial call
    # rem = P
    # min_f = 13
    # max_k = 7 (max possible)
    # curr = []
    
    # Optimization: Try to find a good initial best_max.
    # We can greedily construct a solution.
    # Or just run DFS.
    
    dfs(P, 13, 7, [], 0)
    
    return best_factors
