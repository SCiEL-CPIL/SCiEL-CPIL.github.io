"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s

    # Precompute primes <= 1000 and 1..1000
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    n_primes = len(primes)
    assert n_primes == 168
    xs = np.arange(1, 1001, dtype=np.int64)  # 1..1000

    # Fast scorer: returns largest L s.t. every integer 1..L has count >= 2
    def score(res):
        # res: int array length 168, values 0<=res[p]<p
        counts = np.zeros(1001, dtype=np.int32)
        for r, p in zip(res, primes):
            counts[(xs - r) % p == 0] += 1
        # find first index where counts < 2
        bad = np.nonzero(counts < 2)[0]
        if len(bad) == 0:
            return 1000
        first = bad[0]
        # xs[first] is the first integer failing
        return int(xs[first]) - 1

    # Greedy initialization: assign each prime to cover ceil(i/2)
    def greedy_init():
        res = np.zeros(n_primes, dtype=np.int64)
        for i, p in enumerate(primes):
            target = (i + 2) // 2  # ceil((i+1)/2)
            res[i] = target % p
        return res

    best_res = greedy_init()
    best_score = score(best_res)

    # Simulated annealing with single-prime moves
    rng = np.random.default_rng(42)
    T = 50.0
    T_min = 0.5
    cooling = 0.9999

    while time.time() < deadline:
        # Batch of iterations per time check
        batch_end = min(deadline, time.time() + 1.0)
        while time.time() < batch_end:
            idx = rng.integers(0, n_primes)
            p = primes[idx]
            old_r = best_res[idx]
            new_r = rng.integers(0, p)
            if new_r == old_r:
                continue
            best_res[idx] = new_r
            new_score = score(best_res)
            delta = new_score - best_score
            if delta >= 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                best_score = new_score
            else:
                best_res[idx] = old_r
            T *= cooling
            if T < T_min:
                T = T_min

    return best_res.tolist()
# EVOLVE-BLOCK-END
