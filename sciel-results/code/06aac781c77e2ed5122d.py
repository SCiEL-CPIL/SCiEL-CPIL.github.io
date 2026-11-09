import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Helper to calculate the set of distinct small primes dividing a number
    def get_small_primes(x):
        s = set()
        for p in primes:
            if x % p == 0:
                s.add(p)
        return s
    
    # Helper to evaluate a starting index n
    # Returns the number of distinct small primes across all n+i (i=1..24)
    def evaluate(n):
        all_primes = set()
        for i in range(1, 25):
            all_primes.update(get_small_primes(n + i))
        return len(all_primes)
    
    best_n = 1000
    best_score = evaluate(best_n)
    
    # Strategy 1: Check multiples of the primorial of the first few primes
    # If n is a multiple of 2*3*5=30, then n+? will have many multiples of 2,3,5 nearby.
    # Actually, we want the WINDOW [n+1, n+24] to contain numbers with few distinct primes.
    # A window aligned with a multiple of 30 might have numbers like 30k+1 (prime?), 30k+2 (2*15k+1), etc.
    # It's better to just sample.
    
    # Strategy 2: Random sampling with local refinement
    # The space is 1000..1000000. 
    # We know the minimum possible is at least 2 (since we have 24 numbers, 
    # it's impossible for all to be divisible by only {2} or only {3}, etc., 
    # and likely impossible to have only 2 distinct primes across 24 consecutive ints 
    # unless we are very lucky, e.g., all even and all multiples of 3? 
    # No, consecutive ints cannot all be even. 
    # The distinct primes in the window will likely be around 4-6.
    
    # Let's try to find a window where the union of prime factors is small.
    
    # 1. Check a few known "good" structures or just random spots
    # Random sampling
    num_samples = 5000
    for _ in range(num_samples):
        if time.time() > deadline:
            break
            
        n = random.randint(1000, 1000000)
        score = evaluate(n)
        
        if score < best_score:
            best_score = score
            best_n = n
            
            # Local search around this new best
            # If we found a good one, check neighbors
            for delta in range(-10, 11):
                if time.time() > deadline:
                    break
                cand = n + delta
                if 1000 <= cand <= 1000000:
                    s = evaluate(cand)
                    if s < best_score:
                        best_score = s
                        best_n = cand
    
    # 2. Targeted search around multiples of small primorials
    # Numbers like 2*3*5*7 = 210. 
    # A window centered near a multiple of 210 might have high divisibility by 2,3,5,7.
    # Let's check windows starting near multiples of 210, 630, etc.
    
    step = 210
    # We can't iterate all multiples of 210 up to 1e6 (approx 4760 iterations) if it's fast enough.
    # 4760 * 24 * 9 ops is small. Let's do it if time permits.
    
    if time.time() < deadline - 10.0:
        # Iterate through multiples of 210 in the range
        # n+i for i in 1..24. 
        # If n is a multiple of 210, n+1..n+24 covers residues 1..24 mod 210.
        # This doesn't guarantee low distinct primes, but it's a structured search.
        
        # Better: Just iterate n from 1000 to 1000000 in steps? No, too slow.
        # Let's just do a linear scan if we have enough time, but optimized.
        # Actually, random + local search is usually sufficient for these "minimize distinct factors" tasks
        # because the landscape is relatively flat with occasional dips.
        
        # Let's try a denser random search in the first 100k and last 100k? 
        # No, uniform is fine.
        
        # Let's try to be smarter: 
        # The score is the count of distinct primes. 
        # Minimum possible is 2? 
        # Can we have 24 consecutive integers such that every integer is divisible by 2 or 3?
        # No, there are numbers coprime to 6.
        # Can we have 24 consecutive integers such that every integer is divisible by 2, 3, or 5?
        # The Jacobsthal function g(k) is the maximum gap between integers coprime to the product of first k primes.
        # g(3) for 2,3,5 is 10. So any window of length 10 has a number coprime to 30.
        # g(4) for 2,3,5,7 is 22. So any window of length 22 has a number coprime to 210.
        # g(5) for 2,3,5,7,11 is 34.
        # So in a window of 24, there MUST be at least one number coprime to 2*3*5*7 (210).
        # That number will have prime factors >= 11.
        # So the distinct primes in the window must include at least one prime >= 11.
        # Plus the small primes dividing the other numbers.
        # So minimum distinct primes is likely >= 3 or 4.
        
        # Let's just run the random search longer if we have time.
        for _ in range(20000):
            if time.time() > deadline:
                break
            n = random.randint(1000, 1000000)
            score = evaluate(n)
            if score < best_score:
                best_score = score
                best_n = n
                # Quick local refine
                for delta in range(-5, 6):
                    if time.time() > deadline:
                        break
                    cand = n + delta
                    if 1000 <= cand <= 1000000:
                        s = evaluate(cand)
                        if s < best_score:
                            best_score = s
                            best_n = cand

    return best_n
