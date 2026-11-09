import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    MAX_L = 50000

    def full_evaluate(residues):
        residues_arr = np.array(residues, dtype=np.int64)
        chunk = 10000
        for start in range(1, MAX_L + 1, chunk):
            end = min(start + chunk, MAX_L + 1)
            xs = np.arange(start, end, dtype=np.int64)
            count = np.zeros(end - start, dtype=np.int32)
            for i in range(n_primes):
                count += ((xs - residues_arr[i]) % primes_arr[i] == 0).astype(np.int32)
            bad = np.where(count < 2)[0]
            if len(bad) > 0:
                return start + bad[0] - 1
        return MAX_L

    np.random.seed(42)
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = full_evaluate(best_residues)

    def hill_climb(residues, cur_score, max_iters=2000):
        nonlocal best_residues, best_score
        current = list(residues)
        no_improve = 0
        max_no_improve = 80

        for it in range(max_iters):
            if time.time() > deadline:
                break
            idx = np.random.randint(n_primes)
            p = primes[idx]
            old_r = current[idx]
            found = False
            for trial in range(6):
                new_r = np.random.randint(p)
                if new_r == old_r:
                    continue
                current[idx] = new_r
                new_score = full_evaluate(current)
                if new_score > cur_score:
                    cur_score = new_score
                    found = True
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_residues = list(current)
                    break
                current[idx] = old_r
            if not found:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        return current, cur_score

    for restart in range(12):
        if time.time() > deadline:
            break
        if restart == 0:
            init = list(best_residues)
            init_score = best_score
        else:
            if restart % 3 == 0:
                init = [np.random.randint(p) for p in primes]
            else:
                init = list(best_residues)
                for _ in range(5):
                    idx = np.random.randint(n_primes)
                    init[idx] = np.random.randint(primes[idx])
            init_score = full_evaluate(init)
        current, score = hill_climb(init, init_score, max_iters=1500)
        if score > best_score:
            best_score = score
            best_residues = list(current)

    if time.time() < deadline - 10:
        current, score = hill_climb(best_residues, best_score, max_iters=3000)
        if score > best_score:
            best_score = score
            best_residues = list(current)

    return best_residues
