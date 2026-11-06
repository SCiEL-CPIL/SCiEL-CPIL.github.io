def phi_fast(n, spf, limit):
    if n <= 1:
        return 1
    result = n
    temp = n
    while temp > 1:
        if temp <= limit:
            p = spf[temp]
            if p > 1:
                result -= result // p
                while temp % p == 0:
                    temp //= p
            else:
                # temp is prime? No, spf[temp] == temp if prime.
                # If spf[temp] == temp, p = temp.
                # Then result -= result // p.
                # while temp % p == 0: temp //= p -> temp = 1.
                # So this works.
                pass
        else:
            # temp > limit, so temp is prime
            result -= result // temp
            temp = 1
    return result
