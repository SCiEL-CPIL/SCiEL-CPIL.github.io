import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit
    
    # Generate primes up to N
    # Simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    smooth = {1}
    
    for p in primes:
        # Generate new smooth numbers involving p
        # We iterate over a copy of the current smooth set
        # to avoid modifying it while iterating, 
        # but we also need to include multiples of p times existing smooth numbers
        # effectively, new_smooth = smooth U { x * p^k | x in smooth, k>=1, x*p^k <= N }
        
        # Optimization: only iterate x such that x*p <= N
        # But we can just do the while loop
        
        current_size = len(smooth)
        # We need to add numbers formed by p
        # We can iterate over the original smooth set
        # Note: if we add to smooth while iterating, we might miss combinations?
        # No, if we iterate over a snapshot, we add x*p, x*p^2 etc.
        # But what about (x*p)*p? That is x*p^2, which is covered by the while loop.
        # What about y where y is newly added (e.g. x*p) and we multiply by p again?
        # The while loop handles powers of p for a fixed base x.
        # So iterating over a snapshot of the set BEFORE adding p-multiples is correct.
        
        snapshot = list(smooth)
        for x in snapshot:
            y = x * p
            while y <= N:
                smooth.add(y)
                y *= p
        
        # Check for solution
        # We can check if any a in smooth has N-a in smooth
        # To speed up, we can check smaller a first? Doesn't matter for existence.
        found = False
        for a in smooth:
            if time.time() > deadline:
                # Fallback to best effort or return current best?
                # The task says return valid a,b. If timeout, we should return something valid.
                # But we want the optimal. If we timeout, we might not have found it.
                # However, the problem is small. 123457 is small.
                # The number of smooth numbers is small.
                # Let's assume it finishes quickly.
                pass
            
            b = N - a
            if b in smooth:
                return [a, b]
                
    # If no solution found (should not happen as N itself is a solution if N is smooth? 
    # No, N might not be smooth. But a=1, b=N-1. If N-1 is smooth, we are good.
    # Eventually p will be large enough to cover N-1 or similar)
    # Fallback: return 1, N-1 (might not be optimal but valid)
    return [1, N-1]
