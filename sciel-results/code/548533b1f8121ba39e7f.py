import numpy as np
import time
import sys

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _score(s):
    """Calculate the upper bound on C3, which is the max absolute autocorrelation sidelobe."""
    A = _calc_acf(s)
    # Exclude A[0] which is N
    if len(A) <= 1:
        return 0.0
    return float(np.max(np.abs(A[1:])))

def _legendre_symbol_seq(p, n):
    """
    Generate a sequence of length n based on Legendre symbols modulo p.
    Legendre symbol (a/p) is 1 if a is a quadratic residue mod p, -1 if not, 0 if a=0.
    We map 0 to 1 arbitrarily to keep it a signed sequence.
    """
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Map index i to (i+1) to avoid 0, or just use i+1
        val = (i + 1) % p
        # Compute Legendre symbol
        ls = pow(val, (p - 1) // 2, p)
        if ls == p - 1:
            seq[i] = -1
        else:
            seq[i] = 1
    return seq

def _local_search(s, deadline, rng):
    """
    Perform local search using block flips.
    s: initial sequence
    deadline: absolute time deadline
    rng: numpy random generator
    """
    current_s = s.copy()
    current_score = _score(current_s)
    
    best_s = current_s.copy()
    best_score = current_score
    
    no_improve_count = 0
    max_no_improve = 5000
    step_count = 0
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        step_count += 1
        
        # Choose a random block to flip
        # Block length between 1 and 10
        block_len = rng.integers(1, 11)
        if block_len > N:
            block_len = N
        start_idx = rng.integers(0, N - block_len + 1)
        
        old_block = current_s[start_idx:start_idx+block_len].copy()
        
        # Flip the block
        current_s[start_idx:start_idx+block_len] = -old_block
        
        new_score = _score(current_s)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_s = current_s.copy()
        else:
            # Revert the flip
            current_s[start_idx:start_idx+block_len] = old_block
            no_improve_count += 1
            
            # If stuck for too long, try a larger perturbation or restart
            if no_improve_count % 100 == 0:
                # Randomly flip a few individual bits to jump out of local minimum
                num_flips = rng.integers(1, 5)
                indices = rng.choice(N, size=num_flips, replace=False)
                current_s[indices] = -current_s[indices]
                current_score = _score(current_s)
                if current_score < best_score:
                    best_score = current_score
                    best_s = current_s.copy()
                no_improve_count = 0

    return best_s, best_score

def construct():
    start_time = time.time()
    # Safety margin: finish 5 seconds before the hard limit
    deadline = start_time + 115.0
    
    best_seq = np.ones(N, dtype=np.int8)
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Try Legendre Symbol Sequences
    # Find primes near N
    primes_to_try = [397, 401, 409]
    for p in primes_to_try:
        if time.time() > deadline:
            break
        s_leg = _legendre_symbol_seq(p, N)
        # Evaluate directly first
        val = _score(s_leg)
        if val < best_val:
            best_val = val
            best_seq = s_leg.copy()
        
        # Run local search on it
        s_opt, val_opt = _local_search(s_leg, deadline, rng)
        if val_opt < best_val:
            best_val = val_opt
            best_seq = s_opt.copy()
            
    # 2. Try Random Sequences with Local Search
    num_random_restarts = 50
    for i in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        s_rand = rng.integers(0, 2, size=N, dtype=np.int8) * 2 - 1
        
        # Quick evaluation
        val = _score(s_rand)
        if val < best_val:
            best_val = val
            best_seq = s_rand.copy()
            
        # Run local search
        # Allocate less time for later restarts if we are close to deadline
        # Simple heuristic: just run until deadline or local max
        s_opt, val_opt = _local_search(s_rand, deadline, rng)
        if val_opt < best_val:
            best_val = val_opt
            best_seq = s_opt.copy()
            
    return best_seq.tolist()
