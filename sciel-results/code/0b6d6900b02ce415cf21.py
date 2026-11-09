# EVOLVE-BLOCK-START
import time
import numpy as np
from bisect import bisect_right

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [primes[0]]
    
    max_gap = 10000
    
    # dp[t][s] = length of longest valid subsequence ending at primes[t] 
    #            with immediate predecessor primes[s] (s < t)
    # parent[t][s] = index r < s such that dp[s][r] was used, or -1
    
    # prev_best[s] = array of size (max_gap+1)
    #   prev_best[s][g] = max{dp[s][r] : r < s, primes[s]-primes[r] <= g}
    # argmax_s[s] = array of size (max_gap+1)  
    #   argmax_s[s][g] = the r that achieves prev_best[s][g]
    
    dp = [None] * n       # dp[t] is list of length t
    parent = [None] * n   # parent[t] is list of length t
    prev_best = [None] * n
    argmax_s = [None] * n
    
    global_max = 1
    best_t = 0
    best_s = -1
    
    # Process s from 0 to n-1
    for s in range(n):
        if time.time() - start_time > DEADLINE:
            break
        
        if s == 0:
            dp[s] = []
            parent[s] = []
            # For s=0, no predecessors. prev_best[0][g] = 1 for all g (sequence of just primes[0])
            prev_best[s] = np.ones(max_gap + 1, dtype=np.uint8)
            argmax_s[s] = np.full(max_gap + 1, -1, dtype=np.int16)
            continue
        
        dp_s = [0] * s
        parent_s = [-1] * s
        
        for r in range(s):
            gap_sr = primes[s] - primes[r]
            
            if r == 0:
                dp_s[r] = 2
                parent_s[r] = -1
            else:
                # Query: max dp[r][m] where primes[r]-primes[m] <= gap_sr
                pb_r = prev_best[r]
                if pb_r is not None and gap_sr <= max_gap:
                    val = int(pb_r[gap_sr])
                    if val >= 2:  # val=1 means just primes[r] alone, giving length 2
                        dp_s[r] = 1 + val
                        parent_s[r] = int(argmax_s[r][gap_sr])
                    elif val == 1:
                        # Only primes[r] itself, so sequence is primes[r], primes[s]
                        dp_s[r] = 2
                        parent_s[r] = -1
                    else:
                        dp_s[r] = 2
                        parent_s[r] = -1
                else:
                    dp_s[r] = 2
                    parent_s[r] = -1
            
            if dp_s[r] > global_max:
                global_max = dp_s[r]
                best_t = s
                best_s = r
        
        dp[s] = dp_s
        parent[s] = parent_s
        
        # Build prev_best[s] and argmax_s[s]
        # For each r < s: gap = primes[s]-primes[r], value = dp_s[r]
        # prev_best[s][g] = max dp_s[r] for r < s with gap <= g
        
        # Create sorted list of (gap, value, r)
        entries = []
        for r in range(s):
            gap = primes[s] - primes[r]
            entries.append((gap, dp_s[r], r))
        entries.sort(key=lambda x: x[0])
        
        pb = np.zeros(max_gap + 1, dtype=np.uint8)
        am = np.full(max_gap + 1, -1, dtype=np.int16)
        
        current_max = 0
        current_arg = -1
        entry_idx = 0
        for g in range(1, max_gap + 1):
            while entry_idx < s and entries[entry_idx][0] <= g:
                if entries[entry_idx][1] > current_max:
                    current_max = entries[entry_idx][1]
                    current_arg = entries[entry_idx][2]
                entry_idx += 1
            pb[g] = current_max
            am[g] = current_arg
        
        prev_best[s] = pb
        argmax_s[s] = am
    
    # Reconstruct the best sequence
    if best_s == -1:
        return [primes[best_t]]
    
    seq = [best_t, best_s]
    ct = best_t
    cs = best_s
    
    while parent[ct] is not None and len(parent[ct]) > cs and parent[ct][cs] != -1:
        ns = parent[ct][cs]
        seq.append(ns)
        ct = cs
        cs = ns
    
    seq.reverse()
    result = [primes[i] for i in seq]
    
    # Verify nondecreasing gaps
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        for i in range(len(gaps)-1):
            if gaps[i] > gaps[i+1]:
                # Fallback: return a simple valid sequence
                return [primes[0], primes[1]] if n >= 2 else [primes[0]]
    
    return result

# EVOLVE-BLOCK-END
