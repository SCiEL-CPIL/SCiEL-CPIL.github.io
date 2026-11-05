"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute smallest prime factor (SPF) for all integers up to 10^6
    # This allows O(log n) factorization and thus O(log n) totient computation
    # for any candidate n.
    N = 1000000
    spf = list(range(N + 1))
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Function to compute Euler's totient phi(n) using SPF
    def totient(x):
        result = x
        p = spf[x]
        while p > 1:
            result -= result // p
            while x % p == 0:
                x //= p
            if x > 1:
                p = spf[x]
            else:
                break
        return result

    # Search for the smallest n in [1, N] such that phi(n) is divisible by 120
    for n in range(1, N + 1):
        if totient(n) % 120 == 0:
            return n

    return 1

# EVOLVE-BLOCK-END
