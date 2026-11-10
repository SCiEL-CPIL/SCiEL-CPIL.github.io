"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_prime_sieve(n):
    """Return a boolean array where index i is True if i is prime."""
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    N = 10000
    is_prime = _is_prime_sieve(N)
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n_primes = len(primes)
    
    if n_primes == 0:
        return []
    if n_primes == 1:
        return primes
    
    # DP approach:
    # best[j][g] = max length of valid subsequence ending at primes[j] with last gap exactly g
    # prefix_max[i][g] = max(best[i][1], ..., best[i][g])
    # argmax_prev_gap[i][g] = the largest h <= g that achieves prefix_max[i][g]
    
    max_gap = N  # Max possible gap
    best = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    prefix_max = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    argmax_prev_gap = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
        for i in range(j):
            gap = primes[j] - primes[i]
            if gap <= 0 or gap > max_gap:
                continue
            pm = prefix_max[i, gap]
            if pm > 0:
                candidate = pm + 1
                if candidate > best[j, gap]:
                    best[j, gap] = candidate
                    parent_i[j, gap] = i
                    parent_gap[j, gap] = int(argmax_prev_gap[i, gap])
            else:
                if 2 > best[j, gap]:
                    best[j, gap] = 2
                    parent_i[j, gap] = i
                    parent_gap[j, gap] = 0
        
        # Compute prefix_max[j] and argmax_prev_gap[j]
        if j < n_primes - 1:
            # Vectorized prefix max
            prefix_max[j, :] = np.maximum.accumulate(best[j, :])
            # Compute argmax_prev_gap: largest index h <= g where best[j][h] == prefix_max[j][g]
            # Trick: use cumulative max with a tie-breaking that picks the largest index
            # We create a "score" that is (best_value, index) and take cummax element-wise
            # Since we want the LARGEST index achieving the max, we can use:
            # For each position, if best[j][g] == prefix_max[j][g], it's a candidate.
            # We want the rightmost such position up to g.
            # Efficient approach: use np.where on a running condition
            b = best[j, :]
            pm = prefix_max[j, :]
            # Create a boolean mask where b == pm (these are positions that achieve the prefix max)
            # Then for each g, argmax_prev_gap[j,g] = largest index <= g where mask is True
            # We can do this by: for positions where b==pm, set value to index; where not, set to -inf
            # Then cummax gives us the largest index <= g that achieved the max
            mask = (b == pm)
            indices = np.where(mask, np.arange(max_gap + 1), -1)
            # cummax on indices: but we need the largest index, not just any max
            # np.maximum.accumulate on indices where mask is true, -1 elsewhere
            argmax_prev_gap[j, :] = np.maximum.accumulate(indices)
    
    # Find the best (j, gap)
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n_primes):
        row_max = int(np.max(best[j, :]))
        if row_max > max_len:
            max_len = row_max
            best_gap = int(np.argmax(best[j, :]))
            best_j = j
    
    if max_len < 2 or best_j < 0:
        return [primes[0]]
    
    # Reconstruct the subsequence
    result = []
    j = best_j
    gap = best_gap
    while j >= 0:
        result.append(primes[j])
        pi = parent_i[j, gap]
        pg = parent_gap[j, gap]
        if pi < 0 or pg < 0:
            break
        j = pi
        gap = pg
        if gap == 0:
            result.append(primes[j])
            break
    
    result.reverse()
    
    # Verify the result is valid
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Invalid, return greedy result as fallback
            values = []
            for p in range(2, 10001):
                if is_prime[p]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
            return values
    
    return result

# EVOLVE-BLOCK-END
