import math
import time

def construct():
                start_time = time.time()
                deadline = start_time + 120 # Safety margin
                
                # First 20 primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
                
                best_n = 2 # Fallback
                best_score = 0
                
                # We will search subsets of these primes.
                # To keep it fast, we focus on sizes 15 and 14.
                # Actually, let's just generate all valid square-free products.
                
                candidates = []
                
                # Recursive generation
                def gen(idx, current_prod, current_divs_count, selected_primes):
                    if time.time() > deadline - 5:
                        return
                    if current_prod > 10**18:
                        return
                    # If we have enough primes, evaluate
                    # We can prune if current_prod * product_of_smallest_remaining > 10**18?
                    # Simpler: just add to candidates if size is high?
                    # Let's collect all valid ones, then sort by score?
                    # No, collecting all might be too many if we go up to size 10.
                    # But we know size 15 is best.
                    
                    # Heuristic: Only evaluate if len(selected_primes) >= 10
                    # And generate all subsets up to size 15.
                    
                    pass
                
                # This is getting complicated.
                # Let's just hardcode the top few candidates:
                # 1. P_15
                # 2. P_14 * 47 (which is P_15)
                # 3. P_14 * 53
                # 4. P_14 * 59
                # ...
                
                # Let's generate candidates of size 15 and 14 explicitly.
                
                def get_divisors(prime_list):
                    divs = [1]
                    for p in prime_list:
                        new_divs = [d * p for d in divs]
                        divs.extend(new_divs)
                    return divs
                
                def calc_score(prime_list):
                    if not prime_list:
                        return 0
                    n = 1
                    for p in prime_list:
                        n *= p
                    if n > 10**18:
                        return -1
                    divs = get_divisors(prime_list)
                    divs.sort()
                    score = 0
                    for i in range(len(divs) - 1):
                        if math.gcd(divs[i], divs[i+1]) == 1:
                            score += 1
                    return score
                
                # Generate candidates
                # Size 15
                # We can use itertools.combinations for the first 16 primes.
                # Combinations of 15 from 16 is 16 candidates.
                # Combinations of 15 from 17 is 17 candidates.
                # Let's check combinations of 15 from the first 20 primes.
                # C(20, 15) = C(20, 5) = 15504.
                # This is small enough to check all!
                
                import itertools
                
                best_score_val = 0
                best_n_val = 2
                
                # Check all subsets of size 15 from first 20 primes
                for combo in itertools.combinations(primes, 15):
                    n = 1
                    for p in combo:
                        n *= p
                    if n > 10**18:
                        continue
                    s = calc_score(combo)
                    if s > best_score_val:
                        best_score_val = s
                        best_n_val = n
                        
                # Check all subsets of size 14 from first 20 primes
                # C(20, 14) = C(20, 6) = 38760.
                # Also small enough.
                for combo in itertools.combinations(primes, 14):
                    if time.time() > deadline - 10:
                        break
                    n = 1
                    for p in combo:
                        n *= p
                    if n > 10**18:
                        continue
                    s = calc_score(combo)
                    if s > best_score_val:
                        best_score_val = s
                        best_n_val = n
                        
                return best_n_val
