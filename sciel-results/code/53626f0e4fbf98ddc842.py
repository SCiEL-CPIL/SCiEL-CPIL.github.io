import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Primes to consider. 
    # Product of first 15 primes is ~6.1e17. 
    # Product of first 16 primes (incl 53) is ~3.2e19 > 10^18.
    # So max distinct primes is 15 if all exponents are 1.
    # If we allow exponents > 1, we might use fewer primes or higher exponents on small ones.
    # Let's consider first 15 primes.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    k = len(primes)
    
    # Precompute powers for fast divisor generation?
    # Actually, for scoring, we need sorted divisors and their masks.
    
    # Scoring Function:
    # 1. Generate divisors and their prime masks.
    # 2. Sort divisors by value.
    # 3. Count adjacent pairs with gcd 1 (mask & next_mask == 0).
    
    def score_exponents(exps):
        # Check product limit
        prod = 1
        for i in range(k):
            if exps[i] > 0:
                prod *= primes[i] ** exps[i]
                if prod > 10**18:
                    return -1
        
        # Generate divisors
        # divs_val: list of divisor values
        # divs_mask: list of bitmasks
        divs_val = [1]
        divs_mask = [0]
        
        for i in range(k):
            if exps[i] == 0:
                continue
            p = primes[i]
            bit = 1 << i
            powers = [1]
            for _ in range(exps[i]):
                powers.append(powers[-1] * p)
            
            new_vals = []
            new_masks = []
            for val in divs_val:
                for j in range(len(powers)):
                    new_vals.append(val * powers[j])
                    # If j==0, mask contribution is 0. Else, bit is set.
                    # Wait, if exponent is > 0, the prime is present.
                    # The mask represents *which* primes divide the divisor.
                    # So if j > 0, the prime i is in the divisor.
                    mask_contrib = bit if j > 0 else 0
                    new_masks.append(divs_mask[len(new_vals)//len(powers)] | mask_contrib) # This index logic is messy
            # Let's do it cleaner
            new_vals = []
            new_masks = []
            idx = 0
            for val in divs_val:
                for j in range(len(powers)):
                    new_vals.append(val * powers[j])
                    mask_contrib = bit if j > 0 else 0
                    new_masks.append(divs_mask[idx] | mask_contrib)
                idx += 1
            
            divs_val = new_vals
            divs_mask = new_masks
            
        # Sort by value
        # Create list of (val, mask) and sort
        pairs = list(zip(divs_val, divs_mask))
        pairs.sort(key=lambda x: x[0])
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(pairs) - 1):
            if (pairs[i][1] & pairs[i+1][1]) == 0:
                score += 1
        return score

    # Initial State:
    # Start with the primorial of the first 15 primes (all exps = 1)
    # This is a known good starting point for high divisor count.
    best_exps = [1] * k
    # Check if product is valid
    prod = 1
    for i in range(k):
        prod *= primes[i]
    if prod > 10**18:
        # Fallback to fewer primes if needed, though 15 primes product is ~6e17
        # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
        # Which is < 10^18. So valid.
        pass
    
    best_score = score_exponents(best_exps)
    current_exps = best_exps[:]
    current_score = best_score
    
    # Simulated Annealing
    # Temperature schedule: Start high, decay
    T = 100.0
    T_min = 0.1
    alpha = 0.999 # Decay factor per step? Or time-based?
    # Let's use time-based decay or step-based.
    # Max steps ~ 100,000? 
    # Score calculation is O(tau(n) log tau(n)).
    # For 15 primes, tau = 2^15 = 32768.
    # 32k log 32k is fast.
    # If exponents are higher, tau grows.
    # e.g. 2^10 * 3^5 ... tau could be larger.
    # Let's limit exponents to keep tau manageable.
    # Max exponent for 2: 10^18 / 2^18 ~ 3.8e15? No, 2^60 > 10^18.
    # So max exponent for 2 is 59.
    # But high exponents reduce number of distinct primes.
    
    # Random Seed
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Propose a move
        # 1. Pick a random prime index i
        # 2. Change exponent by -1, 0, or +1 (biased towards 0 or 1?)
        # Let's try:
        # - Increase exponent of a small prime (if product allows)
        # - Decrease exponent of a large prime
        # - Swap: decrease one, increase another
        
        i = random.randint(0, k-1)
        delta = random.choice([-1, 1, 1, 1]) # Bias towards increasing? Or random?
        # Let's try random delta from {-1, 1} mostly, sometimes 0? No, 0 is no change.
        # Let's pick a random prime and a random delta from {-1, 1, 2, -2}
        delta = random.choice([-1, 1, 1, 2, -2])
        
        new_exps = current_exps[:]
        new_exps[i] += delta
        
        # Validate
        if new_exps[i] < 0:
            continue
            
        # Quick product check
        # Instead of recalculating full product every time, we can track it?
        # Or just do the power calc in score_exponents?
        # score_exponents already checks product.
        
        new_score = score_exponents(new_exps)
        
        if new_score < 0:
            continue # Invalid
        
        # Acceptance Probability
        if new_score >= current_score:
            accept = True
        else:
            # Boltzmann
            # Delta is negative
            delta_s = new_score - current_score
            # If T is small, exp(delta_s/T) is small
            try:
                prob = math.exp(delta_s / T)
            except OverflowError:
                prob = 0.0
            accept = random.random() < prob
        
        if accept:
            current_exps = new_exps
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_exps = current_exps[:]
        
        # Cool down
        # Time-based cooling is more robust
        elapsed = time.time() - start_time
        total_time = 145.0
        # Simple exponential decay based on elapsed time?
        # T = T_min + (T_start - T_min) * exp(-elapsed / tau_cool)
        # Let's just multiply by a factor each step, but ensure we don't go below T_min too fast
        T = max(T_min, T * 0.9995)
        
        # If we have very few steps left, we might want to do a local refinement?
        # For now, SA should handle it.
        
        # Optimization: If current_score is very high, maybe T should drop faster?
        # No, standard schedule is fine.
        
        # Break if T is too low and we've been stuck?
        # SA handles stuckness via T.
        
    # Final Refinement: Local Search on best_exps
    # Try single-step moves from best_exps to see if we can squeeze more
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(k):
            for delta in [-1, 1, 2, -2, 3, -3]:
                new_exps = best_exps[:]
                new_exps[i] += delta
                if new_exps[i] < 0:
                    continue
                s = score_exponents(new_exps)
                if s > best_score:
                    best_score = s
                    best_exps = new_exps
                    improved = True
                    # Restart local search from new best
                    break
            if improved:
                break
                
    # Construct n
    n = 1
    for i in range(k):
        if best_exps[i] > 0:
            n *= primes[i] ** best_exps[i]
            
    return n

# To test locally (commented out for submission)
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     res = construct()
#     end = time.time()
#     print(f"Result: {res}")
#     print(f"Time: {end-start:.2f}s")
#     # Calculate score for debug
#     # (Need to import score_exponents or duplicate logic)
