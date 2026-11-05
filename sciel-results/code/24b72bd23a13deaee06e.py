for p in small_primes:
    p_power = p
    while p_power <= max_val:
        for m in range(p, max_val + 1, p):
            while m % p == 0:
                m //= p
            break  # fallback to loop
    for m in range(p, max_val + 1, p):
        temp = m
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count > 0:
            small_part[m] *= p ** count
