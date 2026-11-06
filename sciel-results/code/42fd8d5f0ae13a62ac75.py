"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def _sieve(n):
    """Return list of primes <= n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    primes = _sieve(10000)
    n = len(primes)
    
    # dp[i] = length of longest valid sequence ending at primes[i]
    # parent[i] = index of previous element in the sequence
    # last_gap[i] = gap between primes[i] and primes[parent[i]]
    dp = [1] * n
    parent = [-1] * n
    last_gap = [0] * n  # gap into the current element (0 for single element)
    
    # threshold[j] = primes[j] + last_gap[j]
    # For i, we need j < i with threshold[j] <= primes[i], maximizing dp[j]
    # Use a Fenwick tree (BIT) for max query over threshold values
    
    # Precompute all possible threshold values and compress
    thresholds = [primes[j] + last_gap[j] for j in range(n)]
    # But last_gap changes as we go... we need to handle this dynamically.
    # Since last_gap[j] is set when we finalize j, and we process in order,
    # we can compute threshold[j] at the time we add j to the structure.
    
    # For the BIT, we need to know the range of threshold values.
    # threshold[j] = primes[j] + last_gap[j] where last_gap[j] <= primes[j] (since gaps are positive and primes are increasing)
    # So threshold[j] <= 2 * primes[j] <= 20000
    # Actually last_gap[j] = primes[j] - primes[parent[j]], so threshold[j] = 2*primes[j] - primes[parent[j]]
    # Max threshold is when parent[j] is small, so threshold[j] < 2*primes[j] <= 20000
    
    max_thresh = 20000
    # BIT for max: bit[k] stores max dp for indices with compressed threshold in range
    # We'll compress threshold values to [1, max_thresh]
    
    # Actually, let's just use a simple array-based approach since n is small.
    # For each i, scan all j < i and check condition. O(n^2) with n=1229 is ~750k ops, very fast.
    # The current approach is already O(n^2) and fast. Let me just optimize the inner loop.
    
    best_idx = 0
    best_len = 1
    
    for i in range(1, n):
        pi = primes[i]
        if time.time() > deadline:
            break
        best_j = -1
        best_dp = 0
        for j in range(i):
            # Check if we can extend from j to i
            if pi - primes[j] >= last_gap[j]:
                if dp[j] > best_dp:
                    best_dp = dp[j]
                    best_j = j
        if best_j != -1:
            dp[i] = best_dp + 1
            parent[i] = best_j
            last_gap[i] = pi - primes[best_j]
        
        if dp[i] > best_len:
            best_len = dp[i]
            best_idx = i
    
    # Reconstruct the sequence
    seq = []
    idx = best_idx
    while idx != -1:
        seq.append(primes[idx])
        idx = parent[idx]
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
