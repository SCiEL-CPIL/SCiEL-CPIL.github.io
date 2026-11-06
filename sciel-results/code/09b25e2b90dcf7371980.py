temp = n_val
for q in primes:
    if q > p:
        break
    while temp % q == 0:
        temp //= q
    if temp == 1:
        break

smooth = (temp == 1)
