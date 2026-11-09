import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    n = len(primes)
    
    # dp[j] is a list of states (gap, length, prev_index, prev_state_idx)
    # prev_index = -1 means start of sequence
    # prev_state_idx = -1 means start of sequence
    dp = [[] for _ in range(n)]
    
    # Initialize first prime
    dp[0] = [(0, 1, -1, -1)]
    
    max_len_global = 1
    best_j = 0
    best_state_idx = 0
    
    for j in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(0, 1, -1, -1)]  # Start new sequence at j
        
        for i in range(j):
            # Check time
            if time.time() > deadline:
                break
                
            pi = primes[i]
            pj = primes[j]
            new_gap = pj - pi
            
            # If new_gap is 0, impossible for distinct primes
            # We iterate over states in dp[i]
            # Optimization: if new_gap is smaller than the min gap in dp[i], 
            # we might still need to check if it's >= some gaps.
            # But dp[i] is sorted by gap and length is increasing.
            # We can't easily break early unless we know all gaps in dp[i] are > new_gap.
            # But we need gap <= new_gap.
            
            for k, (g, l, pi_prev, psi_prev) in enumerate(dp[i]):
                if new_gap >= g:
                    candidates.append((new_gap, l + 1, i, k))
                else:
                    # Since dp[i] is sorted by gap, if current gap > new_gap, 
                    # all subsequent gaps are also > new_gap.
                    # So we can break.
                    break
        
        # Prune candidates
        # Sort by gap, then by length (descending? No, we want to keep max length for each gap prefix)
        # Actually, we want to remove dominated states.
        # State A dominates B if gap_A <= gap_B and len_A >= len_B.
        # Sort by gap ascending.
        candidates.sort(key=lambda x: x[0])
        
        best_states = []
        max_len_so_far = 0
        
        for c in candidates:
            g, l, pi_idx, psi_idx = c
            if l > max_len_so_far:
                best_states.append(c)
                max_len_so_far = l
        
        dp[j] = best_states
        
        if max_len_so_far > max_len_global:
            max_len_global = max_len_so_far
            best_j = j
            # Find the state with this length in dp[j]
            # Since we kept only non-dominated, and lengths are strictly increasing in best_states,
            # the last one has the max length.
            best_state_idx = len(best_states) - 1

    # Reconstruct sequence
    seq = []
    if max_len_global > 0:
        j = best_j
        state_idx = best_state_idx
        while j != -1:
            g, l, prev_j, prev_state_idx = dp[j][state_idx]
            seq.append(primes[j])
            j = prev_j
            state_idx = prev_state_idx
        seq.reverse()
    
    return seq

if __name__ == "__main__":
    construct()
