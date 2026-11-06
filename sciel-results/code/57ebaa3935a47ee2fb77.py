import time
import numpy as np
from heapq import heappush, heappop

def construct():
    start_time = time.time()
    TIME_LIMIT = 80  # Seconds, leave 10s margin

    N_MAX = 1000000
    N_MIN = 10000
    N_PLUS_1 = N_MAX + 1

    # 1. Compute Largest Prime Factor for all numbers up to 1000001
    # Step 1a: Generate primes using vectorized boolean sieve
    is_prime = np.ones(N_PLUS_1 + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N_PLUS_1)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.where(is_prime)[0]

    # Step 1b: Compute LPF by processing primes in increasing order
    # The last prime that divides a number is its largest prime factor
    lpf = np.zeros(N_PLUS_1 + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p

    # 2. Search for optimal n using vectorized operations
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_np1 = lpf[N_MIN+1:N_MAX+2]

    scores = np.maximum(lpf_n, lpf_np1)
    min_score = np.min(scores)
    min_idx = np.argmin(scores)
    best_n = N_MIN + min_idx

    # If we have time, we can try to refine by checking nearby values or using a smooth number approach
    # But since the vectorized scan is O(N) and very fast (< 0.1s), we likely already have the global optimum.
    # The current score is 10.5385, which means the value is 1/10.5385 * seed? No, score = seed/value.
    # Wait, the task says "score is seed/value". The seed program score is 1.0.
    # So if seed program returns value V_seed, score_seed = V_seed / V_seed = 1.0? No, "1.0 equals the seed program".
    # So if my program returns value V_mine, score = V_seed / V_mine.
    # Current score is 10.5385. This implies V_seed / V_mine = 10.5385.
    # So V_mine = V_seed / 10.5385.
    # This means my current program is already significantly better than the seed.
    # To improve further, I need to find a smaller value.
    # The global minimum of max(lpf[n], lpf[n+1]) for n in [10000, 1000000] is what I'm finding.
    # Is it possible there are ties? Yes. But any tie gives the same value, hence same score.
    # So the vectorized scan finds the true global minimum.
    # Therefore, no further search can improve the value, unless I misunderstood the problem.
    # "Minimize the largest prime factor of n(n+1)".
    # largest prime factor of n(n+1) = max(lpf(n), lpf(n+1)).
    # So the vectorized scan is optimal.

    return int(best_n)
