"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_sidelobe_max(f):
    """Compute the max absolute autocorrelation sidelobe (excluding lag 0)."""
    n = len(f)
    max_side = 0.0
    for k in range(1, n):
        ac = np.dot(f[:n-k], f[k:])
        if abs(ac) > max_side:
            max_side = abs(ac)
    return max_side


def _de_bruijn_sequence(n_bits, k=2):
    """Generate a de Bruijn sequence B(k, n_bits) as a list of 0s and 1s."""
    # de Bruijn sequence of order n_bits over alphabet of size k
    # Length is k^n_bits, but we want a cyclic sequence
    # For k=2, n_bits=8 gives length 256
    # For k=2, n_bits=9 gives length 512
    # We need 400, so let's use a different approach
    
    # Actually, let's use the standard de Bruijn sequence generation
    # B(k, n) has length k^n
    # For our purposes, we'll generate a binary de Bruijn sequence and map to {-1, 1}
    
    a = [0] * (k * n_bits)
    sequence = []
    
    def db(t, p):
        if t > n_bits:
            if n_bits % p == 0:
                sequence.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    
    db(1, 1)
    return sequence


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    best_f = None
    best_score = float('inf')
    
    # Strategy 1: De Bruijn based sequence
    # Generate a binary de Bruijn sequence of order 9 (length 512)
    # Then take 400 consecutive elements and map to {-1, 1}
    try:
        db_seq = _de_bruijn_sequence(9, k=2)
        # db_seq has length 512
        # Take 400 elements, map 0->-1, 1->1
        f1 = np.array([2*x - 1 for x in db_seq[:N]], dtype=np.float64)
        score1 = _autocorr_sidelobe_max(f1)
        best_f = f1
        best_score = score1
    except Exception:
        pass
    
    # Strategy 2: Try different offsets in the de Bruijn sequence
    if time.time() < deadline:
        try:
            db_seq = _de_bruijn_sequence(9, k=2)
            seq_len = len(db_seq)
            # Try several offsets
            for offset in range(0, seq_len - N + 1, max(1, (seq_len - N) // 20)):
                f_try = np.array([2*x - 1 for x in db_seq[offset:offset+N]], dtype=np.float64)
                score_try = _autocorr_sidelobe_max(f_try)
                if score_try < best_score:
                    best_score = score_try
                    best_f = f_try
                if time.time() > deadline - 5:
                    break
        except Exception:
            pass
    
    # Strategy 3: Random search with local improvement
    if time.time() < deadline and best_f is not None:
        rng = np.random.RandomState(42)
        current = best_f.copy()
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            # Random flip of a few elements
            n_flips = rng.randint(1, 6)
            idx = rng.choice(N, size=n_flips, replace=False)
            new_f = current.copy()
            new_f[idx] = -new_f[idx]
            new_score = _autocorr_sidelobe_max(new_f)
            if new_score < current_score:
                current = new_f
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_f = current.copy()
            else:
                no_improve_count += 1
    
    # Ensure we return something valid
    if best_f is None:
        return np.ones(N)
    
    return best_f

# EVOLVE-BLOCK-END
