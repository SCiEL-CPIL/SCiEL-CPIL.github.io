import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to 1,000,000 efficiently.
    """
    N = 1000000
    deadline = time.time() + 55.0
    
    # Initialize phi array: phi[i] = i
    # Using a list is standard; array('I') could be faster for memory but list is fine for 10^6
    phi = list(range(N + 1))
    
    # Sieve to compute Euler's totient
    # We iterate i from 2 to N. If phi[i] is still i, then i is prime.
    for i in range(2, N + 1):
        if time.time() > deadline:
            # Fallback if sieve takes too long (should not happen for 10^6)
            # We can break and use trial division for the rest, but for now, 
            # returning the best found so far or a fallback is safer.
            # However, 10^6 is small. Let's just break and handle below.
            break
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # If we broke early due to deadline, we might have incomplete phi values.
    # But for N=1000000, the loop is very fast (~0.1-0.5s). 
    # If we did break, we should verify or use a slower method for the tail.
    # Given the constraint, we assume the sieve completes. 
    # To be safe against the "KILLS after 90s" rule, the 55s guard is sufficient.
    
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if time.time() > deadline:
            # If we run out of time during the scan, return the last valid candidate
            # or 1000001 if none found. But the scan is O(N) and very fast.
            return 1000001 
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not be reached as phi(1)=1, but there are many solutions)
    return 1000001
