import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 123457
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    
    # Function to generate all K-smooth numbers <= N
    def generate_smooth(primes_list, N):
        smooths = set()
        # Recursive generation
        def gen(idx, val):
            if val > N:
                return
            smooths.add(val)
            if idx == len(primes_list):
                return
            p = primes_list[idx]
            # Multiply by p repeatedly
            next_val = val
            while next_val * p <= N:
                next_val *= p
                gen(idx + 1, next_val)
                # Also need to allow skipping this prime? 
                # No, this logic generates numbers where prime idx is used at least once?
                # Standard way:
                # gen(idx, val) considers primes from idx onwards.
                # It should add val.
                # Then for p in primes[idx:], try val*p, val*p^2...
                
        # Better iterative approach or correct recursion
        def gen_correct(idx, val):
            if val > N:
                return
            smooths.add(val)
            if idx == len(primes_list):
                return
            p = primes_list[idx]
            # Option 1: don't use p (move to next prime)
            # Option 2: use p
            # This generates duplicates if not careful.
            # Standard:
            # gen(idx, val) adds val.
            # Then for i in range(len(primes_list)-idx):
            #   p = primes_list[idx+i]
            #   if val*p <= N: gen(idx+i+1, val*p)
            
        # Let's use a simpler BFS/DFS
        def gen(idx, val):
            if val > N:
                return
            smooths.add(val)
            if idx == len(primes_list):
                return
            p = primes_list[idx]
            # Use p
            if val * p <= N:
                gen(idx, val * p)
            # Skip p
            gen(idx + 1, val)
            
        gen(0, 1)
        return smooths

    # Try increasing max prime factor
    # We want to minimize max(P(a), P(b)).
    # This is equivalent to finding the smallest K such that there exist a,b with a+b=N and P(a)<=K, P(b)<=K.
    # This is the same as a,b being K-smooth.
    
    # Start with small primes
    # If we find a solution for K=2, score is huge.
    # If K=3, etc.
    
    # Optimization: Only generate up to N.
    # Check if N-a is in the set.
    
    # Note: P(1)=1. 1 is K-smooth for any K>=1.
    # But a,b must be positive integers.
    
    # Let's try K=2
    # Smooths: 1, 2, 4, 8, ...
    # N=123457 (odd).
    # One even, one odd.
    # Odd 2-smooth is only 1.
    # So a=1, b=123456.
    # Is 123456 2-smooth? No, it has factor 3, 7, 11, 13, 643.
    # So K=2 fails.
    
    # K=3
    # Odd 3-smooth: 1, 3, 9, 27, ...
    # Even 3-smooth: 2, 6, 18, ...
    # We need a+b=N.
    
    # I will iterate through primes list and generate smooth sets.
    # To be safe against time, I will limit the number of primes or check time.
    
    for i in range(len(primes)):
        if time.time() > deadline - 5:
            break
        current_primes = primes[:i+1]
        max_p = current_primes[-1]
        
        smooths = generate_smooth(current_primes, N)
        
        # Check for valid pair
        found = False
        best_a = -1
        # We can iterate through smaller half of smooths to save time?
        # Or just iterate all.
        for a in smooths:
            b = N - a
            if b in smooths:
                # Found a pair
                # We should verify P(a) and P(b) are indeed <= max_p
                # By construction they are.
                # We want to return a, b.
                # To minimize max(P(a), P(b)), we found it for this K.
                # Is it possible that a pair with smaller max P exists?
                # We iterate K in increasing order.
                # So the first K we find is the minimal max P.
                # Wait, the set of K-smooth numbers grows.
                # If a pair exists for K=3, it also exists for K=5.
                # We want the MINIMAL K.
                # So the first successful K is the answer.
                
                # However, we need to be careful.
                # The score depends on the ACTUAL max(P(a), P(b)).
                # If we find a pair for K=7, the actual max P might be 5.
                # But if a pair with max P=5 existed, we would have found it at K=5.
                # So the first K where we find a pair gives the minimal possible max P.
                
                # One detail: P(1)=1.
                # If a=1, P(a)=1.
                # If b is 7-smooth, P(b) <= 7.
                # max(1, P(b)) <= 7.
                
                # So returning this pair is optimal for this K.
                # And since K is minimal, this is globally optimal.
                
                # Just in case, let's pick the pair that minimizes max(P(a), P(b)) within this K?
                # No, all have max P <= K.
                # The "value" is max(P(a), P(b)).
                # If we found a pair at K=7, the value is <= 7.
                # Could it be 5?
                # If it was 5, it would have been found at K=5.
                # Unless the specific pair (a,b) has max P=5, but no OTHER pair with max P=5 exists?
                # No, if a,b have max P=5, they are 5-smooth.
                # So they would be in the set for K=5.
                # So if we didn't find a pair at K=5, no pair with max P=5 exists.
                # Therefore, the pair found at K=7 must have max P=7 (or 6? No, 6 is not prime).
                # So the value is exactly K (if K is prime) or the largest prime factor of the numbers.
                
                # So we can return immediately.
                return [a, b]
                
    # If no small prime solution found, fallback to original method or return best found?
    # The original method found 11.
    # If my loop finishes without finding a solution, it means no solution for K up to last prime.
    # I should include primes up to at least 11 or 13 to ensure I find the 11-solution if it's the best.
    # My prime list goes up to 293.
    # So I will find the 11-solution if it's the minimal one.
    
    # Fallback: Return the 11-solution found by original logic?
    # Or just return [1, N-1]?
    # Better to implement the LPF scan as fallback if the smooth generation fails (shouldn't happen for N=123457).
    
    # Let's add the LPF scan as a backup to guarantee a valid answer.
    # But the smooth generation should find it.
    
    # To be safe, if loop finishes:
    # Run the original LPF logic?
    # It's redundant.
    # I'll just return [1, N-1] as last resort, but it's worse.
    # Actually, I can just let the loop run.
    
    # One issue: `generate_smooth` might be slow for large K?
    # Number of K-smooth numbers is small.
    # For K=11, count is ~ few thousands.
    # For K=100, count is ~ 100k.
    # N=123457.
    # It should be fast.
    
    return [1, N - 1]
