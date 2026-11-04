"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(s):
    """Compute autocorrelation of a 1D array (excluding lag 0)."""
    n = len(s)
    acf = np.zeros(n)
    for lag in range(1, n):
        acf[lag] = np.dot(s[:n-lag], s[lag:])
    return acf

def _score(s):
    """Score: upper bound on C3. We use the max absolute autocorrelation (excluding lag 0)."""
    acf = _compute_acf(s)
    # The C3 bound is likely related to the peak sidelobe
    # Try: max |acf[1:]|
    return np.max(np.abs(acf[1:]))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 values."""
    # Thue-Morse: t(k) = parity of number of 1s in binary representation of k
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    return 1 - 2 * tm  # Convert 0/1 to +1/-1

def _local_search(s, max_iter=50000, deadline=140.0):
    """Local search: flip individual bits to reduce max |acf|."""
    s = s.copy().astype(np.int8)
    n = len(s)
    current_score = _score(s)
    best_score = current_score
    best_s = s.copy()
    
    start = time.time()
    it = 0
    no_improve = 0
    
    while it < max_iter and time.time() < deadline:
        # Pick a random position to flip
        i = np.random.randint(n)
        old_val = s[i]
        s[i] = -s[i]
        
        new_score = _score(s)
        
        if new_score < best_score:
            best_score = new_score
            best_s = s.copy()
            no_improve = 0
        else:
            s[i] = old_val  # revert
            no_improve += 1
        
        it += 1
        
        if no_improve > 5000:
            # Try a random restart with small perturbation
            s = best_s.copy()
            # Flip a few random positions
            for _ in range(3):
                j = np.random.randint(n)
                s[j] = -s[j]
            no_improve = 0
    
    return best_s, best_score

def construct():
    """Construct the signed step function heights."""
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    s0 = _thue_morse(N)
    best_s = s0.copy()
    best_score = _score(s0)
    
    # Local search from Thue-Morse
    s1, score1 = _local_search(s0, max_iter=100000, deadline=deadline)
    if score1 < best_score:
        best_s = s1
        best_score = score1
    
    # Try random initializations if time remains
    if time.time() < deadline - 20:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            s_rand = np.random.choice([-1, 1], size=N).astype(np.int8)
            s2, score2 = _local_search(s_rand, max_iter=50000, deadline=deadline)
            if score2 < best_score:
                best_s = s2
                best_score = score2
    
    # Return as float array (signed step heights)
    return best_s.astype(np.float64)

# EVOLVE-BLOCK-END
