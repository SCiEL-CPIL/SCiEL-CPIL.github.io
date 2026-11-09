"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import log

def _score(U):
    """Compute the score for a sorted list of nonnegative integers containing 0."""
    if not U or U[0] != 0:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U using FFT
    # U is a list of nonnegative integers
    # Create a binary vector of length 2*max_u + 1
    size = 2 * max_u + 1
    v = np.zeros(size, dtype=np.float64)
    for x in U:
        v[x] = 1.0
    
    # FFT convolution for sumset
    f = np.fft.rfft(v, n=2 * size)
    sumset_fft = f * f
    sumset_counts = np.fft.irfft(sumset_fft, n=2 * size)
    sumset_counts = np.round(sumset_counts).astype(np.int64)
    sumset_card = np.sum(sumset_counts > 0)
    
    # Compute difference set U-U
    # For difference set, we need U - U = {a - b : a, b in U}
    # This is equivalent to the autocorrelation of U
    # a - b ranges from -max_u to max_u
    # We can compute this by correlating v with itself
    # Or use the fact that U-U = {d : d = a-b}
    # Let's use FFT: the difference set can be computed as the support of the autocorrelation
    
    # Actually, let's compute it directly for small sets, or use FFT
    # For efficiency, use FFT-based approach:
    # The difference set U-U has elements d where d = a - b for some a,b in U
    # This is the same as the set {d : d + b in U for some b in U}
    # Equivalently, it's the support of the cross-correlation
    
    # Let's use a different approach: shift and multiply
    # For each possible difference d in [-max_u, max_u], check if there exists b in U such that b+d in U
    # This is the autocorrelation
    
    # Using FFT: autocorrelation of v
    # corr[d] = sum_x v[x] * v[x+d]
    # We want to know which d have corr[d] > 0
    
    # The autocorrelation can be computed via FFT:
    # corr = ifft(fft(v) * conj(fft(v)))
    # But we need to handle the circular convolution carefully
    
    # Simpler: just compute it directly if the set is not too large
    # For max_u up to a few thousand, direct computation is O(n^2) which might be slow
    # Let's use FFT-based autocorrelation
    
    # Pad v to avoid circular convolution issues
    # The difference ranges from -max_u to max_u
    # We need a vector of length at least 2*max_u + 1
    # Let's use the standard autocorrelation via FFT
    
    # Create a vector of length 2*size to avoid wrap-around
    v_padded = np.zeros(2 * size, dtype=np.float64)
    v_padded[:size] = v
    
    f2 = np.fft.rfft(v_padded, n=4 * size)
    autocorr_fft = f2 * np.conj(f2)
    autocorr = np.fft.irfft(autocorr_fft, n=4 * size)
    autocorr = np.round(autocorr).astype(np.int64)
    
    # The autocorrelation at lag d (for d >= 0) is autocorr[d]
    # For negative lags, autocorr[-d] = autocorr[d] by symmetry
    # The difference set includes all d in [-max_u, max_u] where autocorr[|d|] > 0
    # Actually, autocorr[d] = sum_x v[x] * v[x+d]
    # So for d >= 0, autocorr[d] > 0 means there exist x, x+d in U, so d is in U-U
    # For d < 0, -d is in U-U iff d is in U-U (by symmetry)
    
    # So diffset_card = 2 * (number of d in [1, max_u] with autocorr[d] > 0) + 1 (for d=0)
    # Wait, autocorr[0] = sum v[x]^2 = n > 0 always
    
    # The positive lags with autocorr[d] > 0 for d in [1, max_u]
    positive_lags = np.sum(autocorr[1:max_u+1] > 0)
    diffset_card = 2 * positive_lags + 1  # +1 for d=0
    
    ratio = diffset_card / sumset_card
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator


def _score_fast(U):
    """Faster score computation for small sets using direct methods."""
    if not U or U[0] != 0:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Direct computation for small sets
    U_set = set(U)
    
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    sumset_card = len(sumset)
    diffset_card = len(diffset)
    
    ratio = diffset_card / sumset_card
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with a good initial set
    # The paper mentions constructions of size 2003 and 54265
    # Let's start with a geometric progression or similar structure
    
    # Initial set: powers of 2 up to some point, plus 0
    # Or start with a small set and grow it
    
    # Let's try a few initial candidates
    candidates = []
    
    # Candidate 1: [0, 1, 3, 7, 15, ...] (Mersenne-like)
    c1 = [0]
    x = 1
    while x < 1000:
        c1.append(x)
        x = 2 * x + 1
    candidates.append(c1)
    
    # Candidate 2: [0, 1, 2, 4, 8, 16, ...] (powers of 2)
    c2 = [0]
    x = 1
    while x < 1000:
        c2.append(x)
        x *= 2
    candidates.append(c2)
    
    # Candidate 3: [0, 1, 3, 9, 27, ...] (powers of 3)
    c3 = [0]
    x = 1
    while x < 1000:
        c3.append(x)
        x *= 3
    candidates.append(c3)
    
    # Candidate 4: [0, 1, 2, 5, 10, 20, ...]
    c4 = [0, 1, 2]
    x = 5
    while x < 1000:
        c4.append(x)
        x *= 2
    candidates.append(c4)
    
    # Evaluate initial candidates
    best_U = None
    best_score = 0.0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score_fast(cand)
        if s > best_score:
            best_score = s
            best_U = list(cand)
    
    if best_U is None:
        return [0, 1, 3]
    
    # Simulated annealing
    current_U = list(best_U)
    current_score = best_score
    best_U = list(current_U)
    best_score = current_score
    
    # Temperature schedule
    T = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    # Precompute some parameters
    max_val_limit = 50000  # Upper bound on max element
    
    while time.time() < deadline:
        # Generate a perturbation
        U_set = set(current_U)
        n = len(current_U)
        max_u = max(current_U) if current_U else 0
        
        # Choose perturbation type
        r = np.random.random()
        
        if r < 0.3 and n > 2:
            # Delete a random element (keep 0)
            deletable = [x for x in current_U if x != 0]
            if deletable:
                x = np.random.choice(deletable)
                new_U = [v for v in current_U if v != x]
        elif r < 0.7:
            # Add a new element
            # Choose a value in [0, max_u + 100] or [0, max_val_limit]
            if max_u < max_val_limit:
                upper = min(max_u + 100, max_val_limit)
                new_x = np.random.randint(0, upper + 1)
                if new_x not in U_set:
                    new_U = current_U + [new_x]
                    new_U.sort()
                else:
                    continue
            else:
                continue
        else:
            # Shift a random element
            shiftable = [x for x in current_U if x != 0]
            if shiftable:
                x = np.random.choice(shiftable)
                delta = np.random.randint(-20, 21)
                new_x = x + delta
                if new_x >= 0 and new_x not in U_set and new_x <= max_val_limit:
                    new_U = [v if v != x else new_x for v in current_U]
                    new_U.sort()
                else:
                    continue
            else:
                continue
        
        # Evaluate new solution
        if time.time() > deadline:
            break
        
        new_score = _score_fast(new_U) if len(new_U) < 200 else _score(new_U)
        
        # Accept or reject
        if new_score >= current_score:
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(new_U)
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta_s = new_score - current_score
            if delta_s > -T * 10:  # Avoid underflow
                prob = np.exp(delta_s / T)
                if np.random.random() < prob:
                    current_U = new_U
                    current_score = new_score
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # If T is very low, try to restart with a different initial set
        if T <= T_min * 1.1:
            # Try a new random initial set
            if time.time() > deadline:
                break
            # Generate a random set
            size = np.random.randint(5, 30)
            new_cand = [0]
            for _ in range(size - 1):
                x = np.random.randint(1, 2000)
                if x not in new_cand:
                    new_cand.append(x)
            new_cand.sort()
            s = _score_fast(new_cand)
            if s > best_score * 0.9:  # Only restart if promising
                current_U = new_cand
                current_score = s
                T = 0.1
    
    return best_U
# EVOLVE-BLOCK-END
