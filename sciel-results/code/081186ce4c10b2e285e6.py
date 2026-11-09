"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(seq):
    """Compute full autocorrelation array for a ±1 sequence."""
    n = len(seq)
    R = np.zeros(n)
    for k in range(n):
        R[k] = np.dot(seq[:n-k], seq[k:])
    return R

def _autocorr_sidelobe_max(R):
    """Compute max absolute autocorrelation sidelobe from R array."""
    if len(R) <= 1:
        return 0.0
    return np.max(np.abs(R[1:]))

def _flip_update_autocorr(R, seq, idx):
    """Update autocorrelation after flipping bit at idx. Returns new R and new seq."""
    n = len(seq)
    new_seq = seq.copy()
    new_seq[idx] = -new_seq[idx]
    
    new_R = R.copy()
    s = seq[idx]  # old value
    
    for k in range(1, n):
        delta = 0.0
        # Term where i = idx: seq[idx] * seq[idx+k]
        if idx + k < n:
            delta += -2.0 * s * seq[idx + k]
        # Term where i+k = idx, i.e., i = idx-k: seq[idx-k] * seq[idx]
        if idx - k >= 0:
            delta += -2.0 * s * seq[idx - k]
        new_R[k] += delta
    
    return new_R, new_seq

def construct():
    deadline = time.time() + 115  # safety margin
    rng = np.random.default_rng(42)
    
    # Initialize with pseudo-random ±1 sequence
    seq = rng.choice([-1, 1], size=N).astype(np.float64)
    R = _compute_autocorr(seq)
    current_score = _autocorr_sidelobe_max(R)
    
    best_seq = seq.copy()
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            new_R, new_seq = _flip_update_autocorr(R, seq, idx)
            new_score = _autocorr_sidelobe_max(new_R)
            if new_score < current_score - 1e-10:
                seq = new_seq
                R = new_R
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_seq = seq.copy()
                improved = True
        if not improved:
            # Random restart with some probability
            if rng.random() < 0.05:
                seq = rng.choice([-1, 1], size=N).astype(np.float64)
                R = _compute_autocorr(seq)
                current_score = _autocorr_sidelobe_max(R)
                if current_score < best_score:
                    best_score = current_score
                    best_seq = seq.copy()
            else:
                break
    
    return best_seq
# EVOLVE-BLOCK-END
