import math
import time

def construct():
        start_time = time.time()
        deadline = start_time + 140  # 10s margin
        
        # Generate first 20 primes
        primes = []
        candidate = 2
        while len(primes) < 20:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        
        best_n = 2
        best_score = 0
        
        # Try square-free numbers with first k primes
        # Also try replacing the largest prime with a larger one?
        # The prompt says "exploiting the remaining product budget to use larger primes".
        # This implies that for a fixed k, we might want to maximize the product?
        # Or maybe the score depends on the specific primes?
        
        # Let's test if the score depends on the primes for k=4.
        # n = 2*3*5*7 = 210. Score 7.
        # n = 2*3*5*11 = 330.
        # Divisors: 1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330.
        # Sorted:
        # 1, 2 (1)
        # 2, 3 (1)
        # 3, 5 (1)
        # 5, 6 (1)
        # 6, 10 (2)
        # 10, 11 (1)
        # 11, 15 (1)
        # 15, 22 (1)
        # 22, 30 (2)
        # 30, 33 (3)
        # 33, 55 (11)
        # 55, 66 (11)
        # 66, 110 (22)
        # 110, 165 (55)
        # 165, 330 (165)
        
        # Coprime:
        # (1,2), (2,3), (3,5), (5,6), (10,11), (11,15), (15,22)
        # Count: 7.
        
        # It seems the score is invariant for square-free numbers with k primes?
        # Let's check k=3.
        # 2*3*5 = 30. Divisors: 1,2,3,5,6,10,15,30.
        # 1,2 (1)
        # 2,3 (1)
        # 3,5 (1)
        # 5,6 (1)
        # 6,10 (2)
        # 10,15 (5)
        # 15,30 (15)
        # Count: 4.
        
        # 2*3*7 = 42. Divisors: 1,2,3,6,7,14,21,42.
        # 1,2 (1)
        # 2,3 (1)
        # 3,6 (3)
        # 6,7 (1)
        # 7,14 (7)
        # 14,21 (7)
        # 21,42 (21)
        # Count: 3.
        
        # Ah! The score DEPENDS on the primes.
        # For 2,3,5 (consecutive small primes), score is 4.
        # For 2,3,7 (gap), score is 3.
        # So we want primes that are "close" together?
        # Or rather, we want the divisors to be interleaved such that adjacent divisors are coprime.
        # In 2,3,5:
        # 1, 2, 3, 5, 6, 10, 15, 30
        # 1-2 (1), 2-3 (1), 3-5 (1), 5-6 (1), 6-10 (no), 10-15 (no), 15-30 (no).
        #
        # In 2,3,7:
        # 1, 2, 3, 6, 7, 14, 21, 42
        # 1-2 (1), 2-3 (1), 3-6 (no), 6-7 (1), 7-14 (no), 14-21 (no), 21-42 (no).
        
        # The "density" of divisors matters.
        # Using smaller primes creates more divisors in the lower range, which might increase coprime adjacencies.
        # So the first k primes is likely optimal.
        
        # Let's verify k=4 with 2,3,5,7 vs 2,3,5,11.
        # We found both had score 7?
        # Let's re-check 2,3,5,11.
        # 1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330
        # 1-2 (1)
        # 2-3 (1)
        # 3-5 (1)
        # 5-6 (1)
        # 6-10 (no)
        # 10-11 (1)
        # 11-15 (1)
        # 15-22 (1)
        # 22-30 (no)
        # 30-33 (no)
        # 33-55 (no)
        # 55-66 (no)
        # 66-110 (no)
        # 110-165 (no)
        # 165-330 (no)
        # Count: 7.
        
        # So for k=4, both give 7.
        # For k=3, 2,3,5 gave 4, 2,3,7 gave 3.
        
        # It seems using the smallest primes is best.
        
        # We will iterate k from 1 to 15.
        # For each k, use the first k primes.
        # Calculate score.
        # Keep track of best n.
        
        for k in range(1, 16):
            if time.time() > deadline:
                break
            
            n = 1
            for i in range(k):
                n *= primes[i]
            
            if n > 10**18:
                break
            
            # Generate divisors
            divisors = [1]
            for p in primes[:k]:
                new_divs = []
                for d in divisors:
                    new_divs.append(d * p)
                divisors.extend(new_divs)
            
            divisors.sort()
            
            score = 0
            for i in range(len(divisors) - 1):
                if math.gcd(divisors[i], divisors[i+1]) == 1:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_n = n
        
        return best_n
