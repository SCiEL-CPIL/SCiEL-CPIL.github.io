"""Program for sumset/difference-set exponent lower bound maximization."""
import time
import numpy as np
from scipy.fft import fft, ifft

DEADLINE = 115.0  # seconds, safety margin under 120s

def _score(U):
    """Compute 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or max(U) == 0:
        return 0.0
    M = max(U)
    n = len(U)
    if n < 2:
        return 0.0
    # Build indicator vector
    size = 2 * M + 1
    # For FFT we need power-of-2 size >= 2*size-1
    fft_size = 1
    while fft_size < 2 * size - 1:
        fft_size <<= 1
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in U:
        vec[x] = 1.0
    # Sumset: convolution of vec with vec -> counts of sums
    f = fft(vec)
    conv_sum = np.real(ifft(f * f))
    # Sumset cardinality: number of positions with count > 0
    # Sums range from 0 to 2M
    sumset_card = int(np.sum(conv_sum[:size] > 0.5))
    # Difference set: convolution of vec with reversed vec
    # rev[i] = vec[M - i] for i in 0..M, 0 otherwise
    # Actually: (U-U) corresponds to cross-correlation
    # Let's use: diff_vec where diff_vec[k] = sum over i of vec[i] * vec[i+k] for k>=0
    # and for k<0, vec[i]*vec[i-k]
    # Simpler: use convolution of vec with reversed vec
    rev = np.zeros(fft_size, dtype=np.float64)
    for x in U:
        rev[M - x] = 1.0  # reversed around M
    f_rev = fft(rev)
    conv_diff = np.real(ifft(f * f_rev))
    # Differences range from -M to M, mapped to indices 0..2M in conv_diff
    # conv_diff[k] = sum_i vec[i] * rev[k-i] = sum_i vec[i] * vec[M-(k-i)] = sum_i vec[i]*vec[M-k+i]
    # Let j = M-k+i, then i = j-M+k... this is getting complex.
    # Alternative: just compute |U-U| directly for small n, use FFT for large n
    # For now, let's do a hybrid: if n is small enough, do O(n^2)
    if n <= 200:
        diffs = set()
        sums = set()
        for a in U:
            for b in U:
                diffs.add(a - b)
                sums.add(a + b)
        sumset_card = len(sums)
        diffset_card = len(diffs)
    else:
        # Use FFT for both
        # For difference set, use the fact that |U-U| = |{a-b: a,b in U}|
        # conv_diff[k] for k in 0..2M gives counts for (M-k) differences... 
        # Let me just do it carefully:
        # We want count of distinct values of a-b for a,b in U
        # Create vector v where v[x]=1 if x in U
        # The autocorrelation: c[d] = sum_x v[x]*v[x+d] for d>=0, and by symmetry c[-d]=c[d]
        # Using FFT: c = ifft(fft(v) * conj(fft(v))) gives circular autocorrelation
        # For linear: need zero-padding
        c = np.real(ifft(f * np.conj(f)))
        # c[d] for d in 0..M-1 is sum_x v[x]*v[x+d] (circular, but with padding it's linear for d < fft_size/2)
        # Actually with fft_size >= 2*M+1, the circular = linear for lags 0..M
        # c[d] > 0 means there exists x with x in U and x+d in U, i.e., d in U-U (positive part)
        # By symmetry, |U-U| = 1 + 2*(number of d in 1..M with c[d] > 0)
        # Wait, that's not right either. c[d] > 0 means d is a positive difference.
        # |U-U| = 1 (for 0) + 2 * (count of d in 1..M where c[d]>0)
        pos_diffs = int(np.sum(c[1:M+1] > 0.5))
        diffset_card = 1 + 2 * pos_diffs
        # For sumset, conv_sum[k] for k in 0..2M
        sumset_card = int(np.sum(conv_sum[:2*M+1] > 0.5))
    
    ratio = diffset_card / max(sumset_card, 1)
    if ratio <= 0:
        return 0.0
    import math
    denom = math.log(2 * M + 1, 2)
    if denom <= 0:
        return 0.0
    val = 1.0 + math.log(ratio, 2) / denom
    return val

def construct():
    start_time = time.time()
    
    # Start with a good seed: Mian-Chowla-like or power-of-2 based
    # The Mian-Chowla sequence: a(1)=1, a(2)=2, a(n+1)=smallest > a(n) s.t. all pairwise sums distinct
    # But for our purpose, we want to maximize |U-U|/|U+U| / log(2maxU+1)
    
    # Start with a geometric-ish set
    U = [0]
    for k in range(1, 12):
        U.append(2**k - 1)
    U.sort()
    
    best_U = list(U)
    best_score = _score(U)
    
    # Local search: try adding/removing/swapping elements
    import random
    random.seed(42)
    
    max_val_limit = 20000  # keep max element reasonable to avoid log penalty
    
    for iteration in range(50000):
        if time.time() - start_time > DEADLINE:
            break
        
        U = list(best_U)
        n = len(U)
        
        # Random move
        move_type = random.random()
        if move_type < 0.4:
            # Add a new element
            candidates = list(range(1, max_val_limit + 1))
            # Bias towards gaps
            if len(U) > 1:
                # Try to add near existing gaps
                U_set = set(U)
                gaps = []
                for i in range(len(U) - 1):
                    gap_start = U[i] + 1
                    gap_end = U[i+1] - 1
                    if gap_start <= gap_end:
                        gaps.extend(range(gap_start, gap_end + 1))
                if gaps:
                    candidates = gaps
                else:
                    candidates = list(range(U[-1] + 1, U[-1] + 500))
            new_val = random.choice(candidates[:min(len(candidates), 5000)])
            if new_val not in U:
                U.append(new_val)
                U.sort()
        elif move_type < 0.7:
            # Remove an element (not 0)
            if n > 2:
                idx = random.randint(1, n - 1)
                U.pop(idx)
        elif move_type < 0.9:
            # Swap an element
            if n > 2:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                new_val = old_val + random.randint(-50, 50)
                new_val = max(1, new_val)
                if new_val not in U:
                    U[idx] = new_val
                    U.sort()
        else:
            # Perturb: change one element randomly
            if n > 2:
                idx = random.randint(1, n - 1)
                new_val = random.randint(1, max_val_limit)
                if new_val not in U:
                    U[idx] = new_val
                    U.sort()
        
        if len(U) < 2 or max(U) == 0:
            continue
        
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    return best_U
