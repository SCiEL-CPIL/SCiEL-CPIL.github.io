"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 140  # 10s safety margin
_w = 0.5 / N  # interval width

def _legendre_symbol(a, p):
    """Compute Legendre symbol (a/p) using Euler's criterion."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p) if pow(a, (p - 1) // 2, p) == 1 else -1

def _make_quadratic_residue_sequence(p, N):
    """Create a sign sequence from quadratic residues mod p, repeated to length N."""
    # Generate Legendre symbols for 0, 1, 2, ..., p-1
    seq = []
    for a in range(p):
        ls = _legendre_symbol(a, p)
        if ls == 0:
            # For a=0, Legendre symbol is 0; assign +1 or -1 arbitrarily
            # Convention: assign -1 to make the sequence more balanced
            seq.append(-1.0)
        else:
            seq.append(float(ls))
    # Tile to length N
    return np.tile(np.array(seq), N // p + 1)[:N]

def _autoconv_l1_bound(h):
    """Compute an upper bound related to the autoconvolution L1 norm."""
    R = np.correlate(h, h, mode='full')
    return _w * np.sum(np.abs(R))

def _greedy_polish(h, max_flips=None):
    """Greedy coordinate descent: flip signs that reduce the bound."""
    h = h.copy()
    current_bound = _autoconv_l1_bound(h)
    
    if max_flips is None:
        max_flips = 5000
    
    flip_count = 0
    improved = True
    
    while improved and flip_count < max_flips:
        if time.time() > _deadline:
            break
        improved = False
        flip_count += 1
        
        # Try flipping each position and see if it improves
        # Use FFT-based evaluation for speed
        for i in range(N):
            if time.time() > _deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            
            new_bound = _autoconv_l1_bound(h)
            
            if new_bound < current_bound - 1e-10:
                current_bound = new_bound
                improved = True
            else:
                h[i] = old_val
    
    return h, current_bound

def construct():
    global _deadline
    
    # Start with quadratic residue sequences for various primes near N
    primes_near_N = []
    for p in range(397, 410):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(np.sqrt(p)) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            primes_near_N.append(p)
    
    candidates = []
    
    # Add quadratic residue sequences
    for p in primes_near_N:
        candidates.append(_make_quadratic_residue_sequence(p, N))
    
    # Also add the seed (all ones) as a fallback
    candidates.append(np.ones(N))
    
    # Evaluate all candidates
    best_h = candidates[0]
    best_score = float('inf')
    
    for h in candidates:
        if time.time() > _deadline:
            break
        try:
            s = _autoconv_l1_bound(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass
    
    # Apply greedy polish to the best candidate
    if time.time() < _deadline - 5:  # Leave 5s safety margin
        best_h, best_score = _greedy_polish(best_h, max_flips=3000)
    
    return best_h
# EVOLVE-BLOCK-END
