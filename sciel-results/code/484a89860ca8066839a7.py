import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s limit

    best_M = 333312
    best_ks = [336, 372, 434]

    def linear_sigma_sieve(N):
        """Compute sigma(n) for all n up to N in O(N) time using linear sieve."""
        sigma = [0] * (N + 1)
        spf = [0] * (N + 1)
        prime_power = [0] * (N + 1)
        sigma_prime_power = [0] * (N + 1)
        primes = []

        sigma[1] = 1
        spf[1] = 1
        prime_power[1] = 1
        sigma_prime_power[1] = 1

        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                prime_power[i] = i
                sigma_prime_power[i] = 1 + i
                sigma[i] = 1 + i
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
                if p == spf[i]:
                    prime_power[i * p] = prime_power[i] * p
                    sigma_prime_power[i * p] = sigma_prime_power[i] + prime_power[i] * p
                    sigma[i * p] = sigma[i] // sigma_prime_power[i] * sigma_prime_power[i * p]
                    break
                else:
                    prime_power[i * p] = p
                    sigma_prime_power[i * p] = 1 + p
                    sigma[i * p] = sigma[i] * (1 + p)

        return sigma

    def find_best(N):
        sigma = linear_sigma_sieve(N)
        m_to_k = {}
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val > 10**18:
                continue
            if M_val not in m_to_k:
                m_to_k[M_val] = []
            m_to_k[M_val].append(k)

        best_M_local = 0
        best_ks_local = []
        for M_val, ks in m_to_k.items():
            valid_ks = [k for k in ks if k <= math.isqrt(M_val)]
            if len(valid_ks) > len(best_ks_local):
                best_ks_local = valid_ks
                best_M_local = M_val
        return best_M_local, best_ks_local

    for N in [200000, 500000, 1000000, 2000000, 5000000, 10000000]:
        if time.time() - start_time > DEADLINE:
            break
        M, ks = find_best(N)
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            if len(best_ks) >= 10:
                break

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
