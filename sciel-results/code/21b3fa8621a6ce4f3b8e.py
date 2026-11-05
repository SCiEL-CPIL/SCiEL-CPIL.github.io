def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We are looking for the minimum n such that 120 | phi(n).
    120 = 2^3 * 3 * 5.
    
    Checking small values:
    - Primes p: phi(p) = p-1. We need p = 120k + 1.
      Smallest prime p = 241 (since 121=11^2, 241 is prime). phi(241)=240, div by 120.
    - Composites n = p^a q^b ...
      phi(n) = product p_i^(a_i-1) (p_i-1).
      We need the product to be divisible by 2^3 * 3 * 5.
      
      Let's check n < 143.
      If n is square-free, n = p q. phi(n) = (p-1)(q-1).
      We need (p-1)(q-1) div by 120.
      Possible pairs (p,q) with p<q, p*q < 143:
      ... (see detailed analysis in previous versions) ...
      (11,13): phi=120. YES. n=11*13=143.
      
      It seems 143 is the global minimum.
      
    Implementation:
    We perform a direct totient sieve up to a small limit (500) to verify the minimality
    of 143 and catch any smaller candidate we might have missed in manual analysis.
    This is extremely fast and robust. If no candidate is found below the limit (which
    shouldn't happen since 143 < 500), we fall back to 143.
    """
    limit = 500
    # phi[i] will store phi(i)
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback if not found in small range (unlikely)
    # But we know 143 is the answer.
    return 143
