def construct():
    N = 123457
    
    # Helper to generate smooth numbers up to N using the given primes
    def get_smooth_numbers(primes, limit):
        smooths = [1]
        for p in primes:
            new_smooths = []
            for s in smooths:
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooths.append(val)
            smooths.extend(new_smooths)
        return set(smooths)
    
    # Helper to create a boolean array for fast lookup
    def get_smooth_bool_array(primes, limit):
        is_smooth = bytearray(limit + 1)
        is_smooth[1] = 1
        smooths = get_smooth_numbers(primes, limit)
        for s in smooths:
            is_smooth[s] = 1
        return is_smooth
    
    # Check prime limits in increasing order
    prime_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for p_max in prime_list:
        primes_up_to = [p for p in prime_list if p <= p_max]
        is_smooth = get_smooth_bool_array(primes_up_to, N)
        
        # Linear scan for the first a where both a and N-a are smooth
        for a in range(1, N // 2 + 1):
            if is_smooth[a] and is_smooth[N - a]:
                return [a, N - a]
    
    return [1, N - 1]
