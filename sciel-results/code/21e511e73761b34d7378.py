import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate primes up to 128
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(128)
    
    selected_numbers = []
    
    for p in primes:
        # Find max exponent
        max_exp = 0
        val = p
        while val <= 128:
            max_exp += 1
            val *= p
            
        # Greedy selection of superincreasing exponents
        # We want a set of exponents e_1 < e_2 < ... such that e_i > sum(e_1...e_{i-1})
        # This guarantees all subset sums are distinct.
        exponents = []
        current_sum = 0
        for e in range(1, max_exp + 1):
            if e > current_sum:
                exponents.append(e)
                current_sum += e
            
        # Add p^e for each selected exponent
        for e in exponents:
            selected_numbers.append(p ** e)
            
    # Sort to ensure strictly increasing
    selected_numbers.sort()
    
    # Verify distinctness of all block products (for safety, though logic above is sound)
    # If timeout is a concern, this verification is O(N^2 * log V). N ~ 40. 1600 ops. Fast.
    if len(selected_numbers) > 0:
        prods = set()
        valid = True
        for i in range(len(selected_numbers)):
            prod = 1
            for j in range(i, len(selected_numbers)):
                prod *= selected_numbers[j]
                if prod in prods:
                    valid = False
                    break
                prods.add(prod)
            if not valid:
                break
        
        # If invalid (should not happen), fall back to greedy prime-only
        if not valid:
             # Fallback: just primes
             return primes
             
    return selected_numbers
