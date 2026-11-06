"""Combined: multi-start greedy hill-climbing with plateau-walking and bottleneck targeting."""
import numpy as np
import time
import random
import math

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    primes = _sieve_primes(1000)
    n_primes = len(primes)

    # Search range: aim for L up to 50,000
    N = 50000

    best_residues = None
    best_score = 0

    rng = np.random.default_rng(42)
    random.seed(42)

    def build_counts(residues):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    def get_score(counts):
        """Find largest L such that counts[x] >= 2 for all x in 1..L."""
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    # Diverse initialization strategies
    initial_strategies = [
        lambda i, p: int(rng.integers(0, p)),           # Random
        lambda i, p: 1 % p,                              # All r_p = 1
        lambda i, p: (p // 2) % p,                       # Midpoint
        lambda i, p: (i * 13 + 5) % p,                   # Linear pattern 1
        lambda i, p: (i * 7 + 3) % p,                    # Linear pattern 2
        lambda i, p: (i * 31 + 11) % p,                  # Linear pattern 3
        lambda i, p: (i * 17 + 7) % p,                   # Linear pattern 4
    ]

    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break

        residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = build_counts(residues)
        current_score = get_score(counts)

        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()

        # Hill climbing with plateau-walking and bottleneck targeting
        stall_count = 0
        max_stall = 3000
        T = 1.0  # Temperature for occasional non-improving moves

        while time.time() < deadline - 5.0:
            # Pick a random prime to mutate
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])

            # Pick new residue with bottleneck targeting bias
            if best_score < N and rng.random() < 0.25:
                # Target the bottleneck: try to cover the first uncovered point
                bottleneck = best_score + 1
                if 1 <= bottleneck <= N:
                    r_new = bottleneck % p
                else:
                    r_new = int(rng.integers(0, p))
            else:
                r_new = int(rng.integers(0, p))

            if r_old == r_new:
                continue

            # Update counts: subtract old
            if r_old == 0:
                if p <= N:
                    counts[p::p] -= 1
            else:
                if r_old <= N:
                    counts[r_old::p] -= 1

            # Add new
            if r_new == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r_new <= N:
                    counts[r_new::p] += 1

            new_score = get_score(counts)
            delta = new_score - current_score

            # Acceptance criterion:
            # - Always accept improvements
            # - Accept equal-score moves with probability 0.5 (plateau walking)
            # - Accept worsening moves with small probability (to escape local minima)
            if delta > 0:
                accepted = True
            elif delta == 0:
                accepted = rng.random() < 0.5
            else:
                # Accept worsening moves with small probability
                accepted = rng.random() < 0.05

            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0

                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert counts
                if r_new == 0:
                    if p <= N:
                        counts[p::p] -= 1
                else:
                    if r_new <= N:
                        counts[r_new::p] -= 1

                if r_old == 0:
                    if p <= N:
                        counts[p::p] += 1
                else:
                    if r_old <= N:
                        counts[r_old::p] += 1

                stall_count += 1
                if stall_count > max_stall:
                    break  # Restart with new initialization

    # Final verification: check if we can go beyond N up to 100,000
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1

        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1

    return best_residues.tolist()
