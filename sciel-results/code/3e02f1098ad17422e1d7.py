max_k = LIMIT // p_pow
for k in range(1, max_k + 1):
    if k % p != 0:
        powerful_part[k * p_pow] *= p_pow
