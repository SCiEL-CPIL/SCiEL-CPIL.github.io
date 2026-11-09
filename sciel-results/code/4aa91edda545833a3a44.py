from math import factorial
from itertools import combinations # Not efficient for search, use DFS

def construct():
        N = factorial(12)
        
        # Get all divisors of N that are > 12
        divisors = []
        for i in range(13, int(N**0.5) + 1):
            if N % i == 0:
                divisors.append(i)
                other = N // i
                if other > 12 and other != i:
                    divisors.append(other)
        # Also check if N itself is a divisor (it is) and > 12
        if N > 12:
             divisors.append(N)
        # Unique and sort
        divisors = sorted(set(divisors))
        
        best_solution = None
        best_max = float('inf')
        
        # We will search for partitions of size k from 1 to 12
        # To minimize the largest factor, we prefer smaller factors earlier?
        # The largest factor is the last one in the sorted sequence.
        # So we want to find a sequence f1 < f2 < ... < fk such that prod = N.
        # Minimize fk.
        
        def dfs(remaining, start_idx, count_left, current_factors, current_max):
            nonlocal best_solution, best_max
            
            if count_left == 0:
                if remaining == 1:
                    if current_max < best_max:
                        best_max = current_max
                        best_solution = list(current_factors)
                return
            
            # Lower bound for the next factor
            # The product of the next 'count_left' factors must be 'remaining'.
            # Since they are distinct and > previous, the smallest possible product
            # for 'count_left' factors starting from min_val is:
            # min_val * (min_val+1) * ...
            # But they must be divisors.
            
            # Pruning:
            # If the product of the smallest 'count_left' available divisors > remaining, prune.
            # This is hard to compute on the fly without precomputation.
            # Simpler pruning:
            # The next factor f must be such that remaining/f can be formed by count_left-1 factors > f.
            # The minimum product for count_left-1 factors > f is roughly f^(count_left-1) * (count_left-1)! ? No.
            # Let's just iterate through possible next factors.
            
            # To minimize the max, we should try to keep the factors as small as possible?
            # No, we want the LAST factor to be small.
            # This implies the earlier factors should be as large as possible?
            # Wait. If f1 * f2 * fk = N.
            # If we make f1 small, fk must be large.
            # If we make f1 large, fk can be smaller?
            # Example: 12! = 13 * 368462... vs 100 * ...
            # Actually, to minimize the max, we want the factors to be balanced.
            # But we are constrained to distinct integers > 12.
            
            # Let's just do a standard DFS that explores all valid partitions and keeps the one with min max.
            # Given the small number of divisors, this should be fast enough.
            
            # Optimization:
            # If current_max >= best_max, we can still continue if the remaining factors might be smaller?
            # No, since we pick in increasing order, the next factors will be >= current_max (actually > last factor >= current_max).
            # So if current_max >= best_max, any new solution will have max >= current_max >= best_max.
            # So we can prune if current_max >= best_max.
            # BUT, current_max is the max of factors chosen SO FAR.
            # The final max will be the last factor, which is >= current_max.
            # So yes, if current_max >= best_max, we can prune this branch.
            
            if current_max >= best_max:
                return

            # Iterate through divisors starting from start_idx
            # The next factor f must be > divisors[start_idx - 1] (if start_idx > 0)
            # And f must divide remaining.
            
            # We can precompute divisors of 'remaining' that are in our list?
            # Or just check divisibility.
            
            # To improve speed, we can estimate the lower bound for the next factor.
            # Let f be the next factor.
            # We need count_left - 1 more factors, all > f.
            # The smallest possible product for these remaining factors is:
            # (f+1) * (f+2) * ... * (f + count_left - 1)
            # So f * (f+1) * ... * (f + count_left - 1) <= remaining
            # This gives a lower bound for f.
            
            min_f = divisors[start_idx] if start_idx < len(divisors) else float('inf')
            
            # Refine min_f using the product constraint
            # Let's just iterate from start_idx
            for i in range(start_idx, len(divisors)):
                f = divisors[i]
                if f > remaining:
                    break
                if remaining % f != 0:
                    continue
                
                # Check if remaining / f can be formed by count_left - 1 factors > f
                rem = remaining // f
                if count_left - 1 == 0:
                    if rem == 1:
                        new_max = max(current_max, f)
                        if new_max < best_max:
                            best_max = new_max
                            best_solution = current_factors + [f]
                    continue
                
                # Pruning:
                # The smallest possible product for count_left-1 factors > f
                # is approximated by f^(count_left-1) * (count_left-1)! ? No.
                # It is (f+1)(f+2)...(f+count_left-1).
                # If rem < (f+1)(f+2)...(f+count_left-1), then impossible.
                min_prod = 1
                for j in range(1, count_left):
                    min_prod *= (f + j)
                    if min_prod > rem:
                        break
                if min_prod > rem:
                    # Since f is increasing, this will only get worse?
                    # Not necessarily, but if f is small, min_prod is small.
                    # If f is large, min_prod is large.
                    # So if min_prod > rem, larger f will make min_prod even larger.
                    # So we can break?
                    # Wait, if f increases, the required rem = N / prod_so_far decreases?
                    # No, we are inside the loop for a specific 'remaining'.
                    # If we increase f, rem decreases.
                    # But the min_prod for the *next* factors increases.
                    # So it's a tradeoff.
                    # However, if min_prod > rem, it means we can't fit count_left-1 distinct integers > f into rem.
                    # If we pick a larger f', rem' = rem * f / f' < rem.
                    # And min_prod' > min_prod.
                    # So if min_prod > rem, then min_prod' > rem' is likely.
                    # Actually, if (f+1)...(f+k-1) > rem, then for any f' > f,
                    # (f'+1)...(f'+k-1) > (f+1)...(f+k-1) > rem.
                    # And rem' < rem.
                    # So yes, we can break.
                    break

                # Recursive call
                dfs(rem, i + 1, count_left - 1, current_factors + [f], max(current_max, f))

        # Try all k from 1 to 12
        # Note: k=1 means [12!], max = 12!
        # We want to minimize max.
        
        for k in range(1, 13):
            # Initial max is inf
            dfs(N, 0, k, [], 0)
            
        if best_solution is None:
            # Fallback: just return [12!]
            return [N]
            
        return best_solution
