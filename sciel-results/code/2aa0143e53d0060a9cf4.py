import time
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i] is a list of tuples (length, last_gap, prev_idx, prev_state_idx)
    # representing non-dominated states ending at primes[i]
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    dp = [None] * n
    
    # Initialize
    dp[0] = [(1, 0, -1, -1)]
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    # For each prime index i, we maintain a list of non-dominated states.
    # When processing prime i, we need to find all states from previous primes j < i
    # where the last gap G <= primes[i] - primes[j].
    
    # Optimization: Instead of checking all j < i for each i, we can use the fact that
    # as i increases, primes[i] increases, so the gap primes[i] - primes[j] increases for fixed j.
    # We can precompute for each state (j, state_idx) the minimum i such that the gap constraint is met.
    
    # Alternative approach: Process states in order of their "required next gap".
    # A state at prime j with last gap G can be extended to prime i if primes[i] - primes[j] >= G.
    # So the minimum prime i that can extend it is the smallest i > j with primes[i] >= primes[j] + G.
    
    # Let's use a priority queue approach:
    # - Maintain a heap of (required_min_prime_value, j, state_idx) for all active states
    # - As we process each prime i in order, pop all states whose required_min_prime_value <= primes[i]
    # - For each popped state, we can extend it to prime i
    
    # But we need to be careful: a state can be extended to multiple future primes.
    # Actually, once a state is "unlocked" (its gap constraint is met for prime i), 
    # it remains valid for all future primes i' > i as well (since gaps only increase).
    
    # So: for each state (j, G, L, ...), compute the first prime index i_min > j such that 
    # primes[i_min] - primes[j] >= G. Then this state becomes "active" starting from i_min.
    
    # We can precompute for each state its activation index, then use a sweep.
    
    # Let's try a different optimization: for each prime i, instead of checking all j < i,
    # we can binary search or use the structure of non-dominated states.
    
    # Key insight: For a fixed i, we want states from j < i where G <= primes[i] - primes[j].
    # Since primes is increasing, primes[i] - primes[j] decreases as j increases (for fixed i).
    # So for larger j, the required gap is smaller, making it easier to satisfy.
    
    # Let's just optimize the inner loop by early termination and better data structures.
    
    # Actually, let me try a cleaner DP with better pruning.
    
    # For each i, dp[i] contains non-dominated (L, G, prev_i, prev_state_idx) states.
    # When building dp[i], for each j < i, we check all states in dp[j].
    # The total work is sum over i of (i * avg_states_per_prime).
    
    # With n ≈ 1229 primes, this is at most ~1229^2/2 * avg_states ≈ 755k * avg_states.
    # If avg_states is small (say < 10), this is ~7.5M operations, which should be fast.
    
    # Let's just make sure the current approach is efficient and add a time guard.
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(1, 0, -1, -1)]
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # For states in dp[j], we need G <= gap
            # dp[j] is sorted by (-L, G), so states are in descending L, ascending G within same L
            # We can break early if G > gap for all remaining states? Not directly, since L varies.
            # But if we process in order of increasing G, we can break when G > gap.
            # Let's re-sort dp[j] by G for this purpose? That's expensive.
            # Alternative: just check all states in dp[j].
            for k, (L, G, pi, si) in enumerate(dp[j]):
                if G <= gap:
                    candidates.append((L + 1, gap, j, k))
        
        # Prune candidates to keep only non-dominated states
        # Sort by length descending, then by gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_gap_encountered = float('inf')
        for cand in candidates:
            L, G, pi, si = cand
            if G < min_gap_encountered:
                non_dom.append(cand)
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        # Update global best
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0

    # Reconstruct the subsequence
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
