import numpy as np
import time

N = 400

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) for prime p."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)
    # This returns 1 if quadratic residue, p-1 (i.e., -1 mod p) if not

def _make_legendre_sequence(p, offset=0):
    """Create a ±1 sequence of length N using Legendre symbols mod p."""
    seq = np.zeros(N, dtype=np.float64)
    for i in range(N):
        x = (i + offset) % p
        if x == 0:
            seq[i] = 1.0  # Convention for 0
        else:
            ls = pow(x, (p - 1) // 2, p)
            seq[i] = 1.0 if ls == 1 else -1.0
    return seq

def _autocorr_max_abs(seq):
    """Compute max absolute autocorrelation value excluding lag 0."""
    n = len(seq)
    max_val = 0.0
    for k in range(1, n):
        c = np.dot(seq[:n-k], seq[k:])
        abs_c = abs(c)
        if abs_c > max_val:
            max_val = abs_c
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_seq = None
    best_score = float('inf')
    
    # Try several primes near N
    primes_to_try = [397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467]
    
    # Also try random sequences as baseline
    np.random.seed(42)
    
    # First, try Legendre symbol sequences
    for p in primes_to_try:
        for offset in range(min(p, 20)):
            seq = _make_legendre_sequence(p, offset)
            score = _autocorr_max_abs(seq)
            if score < best_score:
                best_score = score
                best_seq = seq.copy()
        
        if time.time() > deadline:
            break
    
    # If we have time, try local search on the best sequence
    if best_seq is not None and time.time() < deadline - 30:
        current_seq = best_seq.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Pick a random position to flip
            idx = np.random.randint(N)
            old_val = current_seq[idx]
            current_seq[idx] = -old_val
            
            new_score = _autocorr_max_abs(current_seq)
            
            if new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_seq = current_seq.copy()
                no_improve_count = 0
            else:
                current_seq[idx] = old_val
                no_improve_count += 1
    
    if best_seq is None:
        # Fallback: random sequence
        best_seq = np.random.choice([-1.0, 1.0], N)
    
    return best_seq
