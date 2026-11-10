import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = [False] * len(is_prime[i*i::i])
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000)
    n_primes = len(primes)
    L_CHECK = 5000

    def init_cov(res):
        cov = np.zeros(L_CHECK + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            s = p if r == 0 else r
            if s <= L_CHECK:
                cov[s::p] += 1
        return cov

    def score_of(cov):
        bad = np.where(cov[1:L_CHECK+1] < 2)[0]
        return int(bad[0]) if len(bad) else L_CHECK

    best = 0
    best_res = None

    for restart in range(15):
        if time.time() > DEADLINE - 5:
            break

        if restart == 0:
            res = np.zeros(n_primes, dtype=np.int32)
        elif restart == 1:
            res = np.array([min(1, p-1) for p in primes], dtype=np.int32)
        else:
            res = np.array([np.random.randint(0, primes[i]) for i in range(n_primes)], dtype=np.int32)

        cov = init_cov(res)
        sc = score_of(cov)

        if sc > best:
            best = sc
            best_res = res.copy()

        for it in range(150):
            if time.time() > DEADLINE - 3:
                break
            if sc >= L_CHECK:
                break

            xb = sc + 1
            improved = False

            for i in range(n_primes):
                if time.time() > DEADLINE - 3:
                    break
                p = primes[i]
                nr = xb % p
                if res[i] == nr:
                    continue

                or_ = res[i]
                os_ = p if or_ == 0 else or_
                ns_ = p if nr == 0 else nr

                if os_ <= L_CHECK:
                    cov[os_::p] -= 1
                if ns_ <= L_CHECK:
                    cov[ns_::p] += 1

                nsc = score_of(cov)

                if ns_ <= L_CHECK:
                    cov[ns_::p] -= 1
                if os_ <= L_CHECK:
                    cov[os_::p] += 1

                if nsc > sc:
                    res[i] = nr
                    sc = nsc
                    improved = True
                    break

            if not improved:
                i = np.random.randint(n_primes)
                or_ = res[i]
                p = primes[i]
                res[i] = np.random.randint(0, p)
                os_ = p if or_ == 0 else or_
                ns_ = p if res[i] == 0 else res[i]
                if os_ <= L_CHECK:
                    cov[os_::p] -= 1
                if ns_ <= L_CHECK:
                    cov[ns_::p] += 1
                sc = score_of(cov)

        if sc > best:
            best = sc
            best_res = res.copy()

    if best_res is None:
        best_res = np.zeros(n_primes, dtype=np.int32)

    return best_res.tolist()
