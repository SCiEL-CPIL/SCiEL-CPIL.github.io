import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_CAP = 100000

    window_size = 10000

    def rebuild_counts(residues, size):
        counts = np.zeros(size + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= size:
                counts[start::p] += 1
        return counts

    def find_L(counts, size):
        if size <= 0:
            return 0
        seg = counts[1:size+1]
        if seg.size == 0:
            return 0
        if np.all(seg >= 2):
            return size
        bad = np.where(seg < 2)[0]
        return int(bad[0])

    # Phase 1: Greedy initialization
    residues = [0] * n_primes
    counts = np.zeros(window_size + 1, dtype=np.int32)
    current_L = 0

    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_L = -1

        if p <= 50:
            candidates = list(range(p))
        else:
            candidates = list(range(min(p, 400)))

        for r in candidates:
            start = r if r > 0 else p
            if start > window_size:
                continue
            pos = np.arange(start, window_size + 1, p)
            old_vals = counts[pos].copy()
            counts[pos] += 1
            new_L = find_L(counts, window_size)
            counts[pos] = old_vals
            if new_L > best_L:
                best_L = new_L
                best_r = r

        residues[idx] = best_r
        r = best_r
        start = r if r > 0 else p
        if start <= window_size:
            counts[start::p] += 1
        current_L = find_L(counts, window_size)

    best_L = current_L
    best_residues = residues.copy()

    # Phase 2: Simulated Annealing
    T = 10.0
    cooling_rate = 0.99998
    min_T = 0.05

    while time.time() < deadline:
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]

        # Generate candidate residues
        if p <= 30:
            candidates = [r for r in range(p) if r != old_r]
        else:
            cand_set = set()
            # Targeted: cover near current_L
            for off in range(0, min(10, p)):
                t = (current_L + off) % p
                if t != old_r:
                    cand_set.add(t)
            # Random
            for _ in range(min(15, p)):
                cand_set.add(random.randint(0, p - 1))
            cand_set.discard(old_r)
            candidates = list(cand_set)
            if len(candidates) > 20:
                candidates = random.sample(candidates, 20)
            if not candidates:
                candidates = [random.randint(0, p - 1) for _ in range(5)]

        best_new_L = -1
        best_new_r = old_r

        old_start = old_r if old_r > 0 else p

        for new_r in candidates:
            new_start = new_r if new_r > 0 else p

            if old_start <= window_size:
                counts[old_start::p] -= 1
            if new_start <= window_size:
                counts[new_start::p] += 1

            new_L = find_L(counts, window_size)

            if new_start <= window_size:
                counts[new_start::p] -= 1
            if old_start <= window_size:
                counts[old_start::p] += 1

            if new_L > best_new_L:
                best_new_L = new_L
                best_new_r = new_r

        # Apply the best candidate
        if best_new_r != old_r:
            new_r = best_new_r
            new_start = new_r if new_r > 0 else p

            if old_start <= window_size:
                counts[old_start::p] -= 1
            if new_start <= window_size:
                counts[new_start::p] += 1
            residues[idx] = new_r

            if best_new_L >= current_L:
                current_L = best_new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                delta = best_new_L - current_L
                accepted = False
                if delta > -300:
                    accepted = random.random() < np.exp(delta / T)
                if accepted:
                    current_L = best_new_L
                else:
                    # Revert
                    if new_start <= window_size:
                        counts[new_start::p] -= 1
                    if old_start <= window_size:
                        counts[old_start::p] += 1
                    residues[idx] = old_r

        # Extend window if we've filled it
        if current_L >= window_size and window_size < MAX_CAP:
            new_size = min(window_size * 2, MAX_CAP)
            window_size = new_size
            counts = rebuild_counts(residues, window_size)
            current_L = find_L(counts, window_size)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()

        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_residues
