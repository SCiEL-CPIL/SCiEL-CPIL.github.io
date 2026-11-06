import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    deadline = time.time() + 55
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n < 2:
        return primes
    
    primes_np = np.array(primes, dtype=np.int64)
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # parent[i][j] = index h such that the subsequence ending at i, j extends the one ending at h, i
    dp = np.zeros((n, n), dtype=np.int32)
    parent = np.full((n, n), -1, dtype=np.int32)
    
    max_gap = 10000
    
    # Iterate i from 0 to n-1
    # For each i, we compute dp[i][j] for all j > i
    # We need dp[h][i] for all h < i, which are already computed in previous iterations
    
    for i in range(n):
        if time.time() > deadline:
            break
        
        # Initialize BIT for this i
        # We need to store (max_length, best_h)
        tree_val = [0] * (max_gap + 2)
        tree_h = [-1] * (max_gap + 2)
        
        # Update BIT with all h < i
        # dp[h][i] is the length of sequence ending at h, i
        for h in range(i):
            g = int(primes_np[i] - primes_np[h])
            val = int(dp[h][i])
            if val == 0:
                continue
            
            # BIT update: point update at index g with value val
            idx = g
            while idx <= max_gap:
                if val > tree_val[idx]:
                    tree_val[idx] = val
                    tree_h[idx] = h
                idx += idx & -idx
        
        # Query BIT for all j > i
        for j in range(i + 1, n):
            g = int(primes_np[j] - primes_np[i])
            
            # BIT query: prefix max up to index g
            best_val = 0
            best_h = -1
            idx = g
            while idx > 0:
                if tree_val[idx] > best_val:
                    best_val = tree_val[idx]
                    best_h = tree_h[idx]
                idx -= idx & -idx
            
            if best_val > 0:
                dp[i][j] = best_val + 1
                parent[i][j] = best_h
            else:
                # No valid predecessor, sequence is just i, j
                dp[i][j] = 2
                parent[i][j] = -1

    # Find the best pair (i, j) maximizing dp[i][j]
    # We only care about i < j
    if n < 2:
        return [primes[0]]
        
    max_val = np.max(dp)
    if max_val == 0:
        return [primes[0]]
        
    idx = np.argmax(dp)
    best_i, best_j = np.unravel_index(idx, (n, n))
    
    # Reconstruct the sequence
    seq = []
    i, j = best_i, best_j
    while True:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            seq.append(primes[i])
            break
        j = i
        i = h
    seq.reverse()
    
    return seq
