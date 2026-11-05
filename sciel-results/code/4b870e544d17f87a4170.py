import time
import numpy as np

def _sieve(n):
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0:2] = b'\x00\x00'
    for p in range(2, int(n ** 0.5) + 1):
        if is_p[p]:
            start = p * p
            is_p[start:n + 1:p] = b'\x00' * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if is_p[i]]

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    
    primes = _sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes

    # For each prime j, states[j] is a list of (length, gap, prev_idx, prev_state_idx)
    states = [[] for _ in range(n)]
    
    # Global best
    best_len = 0
    best_gap = 0
    best_j = -1
    best_state_idx = -1

    # Precompute prime array for faster access
    primes_arr = np.array(primes, dtype=np.int32)
    
    for j in range(n):
        pj = primes[j]
        # Initialize with the sequence of just [pj]
        curr_states = [(1, 0, -1, -1)]
        
        # Vectorize gap computation
        if j > 0:
            gaps_to_j = pj - primes_arr[:j]  # gaps from each previous prime to j
            
            for i in range(j):
                pi = primes[i]
                gap = gaps_to_j[i]
                # For each state in i, see if it can extend to j
                for k in range(len(states[i])):
                    len_i, gap_i, prev_i, prev_k = states[i][k]
                    if gap >= gap_i:
                        new_len = len_i + 1
                        curr_states.append((new_len, gap, i, k))
        
        # Prune curr_states to Pareto optimal
        if len(curr_states) > 1:
            curr_states.sort(key=lambda x: (-x[0], x[1]))
            
            pruned = []
            min_gap_so_far = float('inf')
            for state in curr_states:
                l, g, pi, ki = state
                if g < min_gap_so_far:
                    pruned.append(state)
                    min_gap_so_far = g
            states[j] = pruned
        else:
            states[j] = curr_states
        
        # Update global best
        if states[j]:
            l, g, pi, ki = states[j][0]
            if l > best_len or (l == best_len and g < best_gap):
                best_len = l
                best_gap = g
                best_j = j
                best_state_idx = 0

        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct
    seq = []
    cur_j = best_j
    cur_k = best_state_idx
    while cur_j != -1:
        l, g, prev_j, prev_k = states[cur_j][cur_k]
        seq.append(primes[cur_j])
        cur_j = prev_j
        cur_k = prev_k
    
    seq.reverse()
    
    # Local search: try to extend or improve the sequence
    if time.time() - t0 < DEADLINE - 5.0 and len(seq) > 1:
        seq = _local_search(seq, primes, primes_arr, t0, DEADLINE)
    
    return seq

def _local_search(seq, primes, primes_arr, t0, DEADLINE):
    """Try to improve the sequence by local modifications."""
    best_seq = list(seq)
    best_len = len(best_seq)
    
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    
    def is_valid(s):
        if len(s) < 2:
            return True
        gaps = [s[i+1] - s[i] for i in range(len(s)-1)]
        return all(gaps[i+1] >= gaps[i] for i in range(len(gaps)-1))
    
    # Try replacing individual elements
    for trial in range(50):
        if time.time() - t0 > DEADLINE - 2.0:
            break
        
        improved = False
        for pos in range(len(best_seq)):
            old_val = best_seq[pos]
            # Try nearby primes
            idx = prime_index.get(old_val, -1)
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_idx = idx + delta
                if 0 <= new_idx < len(primes):
                    new_val = primes[new_idx]
                    if new_val == old_val:
                        continue
                    candidate = list(best_seq)
                    candidate[pos] = new_val
                    # Check if still increasing
                    if pos > 0 and candidate[pos] <= candidate[pos-1]:
                        continue
                    if pos < len(candidate) - 1 and candidate[pos] >= candidate[pos+1]:
                        continue
                    if is_valid(candidate):
                        # Check if this is an improvement (same length, but we could try to extend)
                        # For now, just keep if valid (same length)
                        # We'd need to try extending to get improvement
                        pass
        
        # Try extending at the end
        if len(best_seq) > 0:
            last = best_seq[-1]
            last_idx = prime_index.get(last, -1)
            if last_idx >= 0:
                last_gap = best_seq[-1] - best_seq[-2] if len(best_seq) >= 2 else 0
                for new_idx in range(last_idx + 1, min(last_idx + 100, len(primes))):
                    new_val = primes[new_idx]
                    new_gap = new_val - last
                    if new_gap >= last_gap:
                        candidate = best_seq + [new_val]
                        if is_valid(candidate):
                            best_seq = candidate
                            best_len = len(best_seq)
                            improved = True
                            break
        
        if not improved:
            break
    
    return best_seq
