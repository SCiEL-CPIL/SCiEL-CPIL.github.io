def construct():
    # We want to maximize tau_perp(n): number of adjacent pairs in the sorted
    # divisor list with gcd 1.
    #
    # Strategy: use a square-free number with as many distinct primes as possible,
    # but also consider adding small prime powers to increase divisor count
    # while keeping the product <= 10^18.
    #
    # For square-free n with k primes, number of divisors is 2^k.
    # The divisors are all products of subsets. Adjacent coprime pairs occur
    # frequently when the primes are small and densely packed in value.
    #
    # Let's try: take the first m primes, and see if we can fit more by
    # using a mixed strategy. But first, let's just maximize k for square-free.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    
    # Find max k such that product of first k primes <= 10^18
    limit = 10**18
    n = 1
    k = 0
    for p in primes:
        if n * p > limit:
            break
        n *= p
        k += 1
    
    # n is now the product of the first k primes
    # Let's check if we can do better by replacing some large primes with
    # additional copies of smaller primes (making it non-square-free)
    # to increase total divisor count.
    
    # Actually, for tau_perp, having more divisors generally helps.
    # Let's try a different approach: use 2^a * 3^b * 5^c * ... with exponents
    # chosen to maximize divisor count while n <= 10^18.
    
    # Greedy: start with n=1, and at each step multiply by the smallest prime
    # that keeps n <= 10^18, but also consider if adding another factor of a
    # smaller prime (increasing exponent) gives more divisors.
    
    # Simpler: brute force over exponent vectors for small primes.
    # Use primes: 2,3,5,7,11,13,17,19,23,29,31
    # Max exponent for 2: log2(10^18) ~ 59.5, so up to 59
    # But we need product <= 10^18.
    
    # Let's try a recursive search for the best exponent combination.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    limit = 10**18
    
    best_n = 1
    best_divs = 0
    
    def search(idx, current_n, current_divs):
        nonlocal best_n, best_divs
        if idx == len(small_primes):
            if current_divs > best_divs:
                best_divs = current_divs
                best_n = current_n
            return
        p = small_primes[idx]
        # Try exponents from 0 upward while current_n * p^e <= limit
        e = 0
        val = 1
        while current_n * val <= limit:
            search(idx + 1, current_n * val, current_divs * (e + 1))
            e += 1
            val *= p
    
    search(0, 1, 1)
    
    return best_n
