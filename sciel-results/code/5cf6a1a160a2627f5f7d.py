# EVOLVE-BLOCK-START
from math import gcd
from itertools import combinations

def construct():
        """
        Find A, B with 3 distinct integers in [1, 128] maximizing distinct pairwise-coprime sums.
        Approach: Systematic search over parity-constrained sets (All Even A, All Odd B) 
        and (All Odd A, All Even B), using randomization with prime bias and local improvement.
        """
        best_A = [1, 2, 3] # Placeholder
        best_B = [4, 5, 6]
        best_count = 0
        
        def check_pairwise_coprime(nums):
            distinct = list(set(nums))
            n = len(distinct)
            for i in range(n):
                for j in range(i + 1, n):
                    if gcd(distinct[i], distinct[j]) != 1:
                        return False
            return True
        
        def evaluate(A, B):
            sums = [a + b for a in A for b in B]
            distinct_sums = list(set(sums))
            if check_pairwise_coprime(distinct_sums):
                return len(distinct_sums)
            return 0
        
        import random
        random.seed(123)
        
        evens = [i for i in range(1, 129) if i % 2 == 0]
        odds = [i for i in range(1, 129) if i % 2 == 1]
        
        # Generate prime candidates for bias
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0: return False
            return True
        
        prime_evens = [2]
        prime_odds = [i for i in odds if is_prime(i)]
        
        # Create weighted pools: Primes and small numbers are more likely
        # We'll just mix them and sample
        pool_A_even = prime_evens + [i for i in evens if i <= 30]
        pool_B_odd = prime_odds + [i for i in odds if i <= 30]
        
        pool_A_odd = prime_odds + [i for i in odds if i <= 30]
        pool_B_even = prime_evens + [i for i in evens if i <= 30]
        
        # Ensure pools have enough elements
        if len(pool_A_even) < 3: pool_A_even = evens[:3]
        if len(pool_B_odd) < 3: pool_B_odd = odds[:3]
        if len(pool_A_odd) < 3: pool_A_odd = odds[:3]
        if len(pool_B_even) < 3: pool_B_even = evens[:3]

        # Random search
        for _ in range(50000):
            # Case 1: A even, B odd
            if random.random() < 0.5:
                A = random.sample(pool_A_even, 3)
                B = random.sample(pool_B_odd, 3)
            else:
                # Case 2: A odd, B even
                A = random.sample(pool_A_odd, 3)
                B = random.sample(pool_B_even, 3)
                
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            
            count = evaluate(A, B)
            if count > best_count:
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if count == 9:
                    break
                    
        # Local improvement on best found
        # Try swapping elements with neighbors in the full range
        for _ in range(5000):
            improved = False
            # Perturb A
            idx = random.randint(0, 2)
            old_val = best_A[idx]
            # Try a random new value
            new_val = random.randint(1, 128)
            if new_val in best_A and new_val != old_val:
                continue
            A_new = best_A[:]
            A_new[idx] = new_val
            if len(set(A_new)) == 3:
                count = evaluate(A_new, best_B)
                if count > best_count:
                    best_A = A_new
                    best_count = count
                    improved = True
                    if count == 9: break
            
            # Perturb B
            if not improved:
                idx = random.randint(0, 2)
                old_val = best_B[idx]
                new_val = random.randint(1, 128)
                if new_val in best_B and new_val != old_val:
                    continue
                B_new = best_B[:]
                B_new[idx] = new_val
                if len(set(B_new)) == 3:
                    count = evaluate(best_A, B_new)
                    if count > best_count:
                        best_B = B_new
                        best_count = count
                        if count == 9: break

        return [best_A, best_B]
    # EVOLVE-BLOCK-END
