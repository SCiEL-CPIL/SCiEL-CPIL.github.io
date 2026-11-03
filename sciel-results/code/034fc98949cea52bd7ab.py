import numpy as np
import time
import math
from itertools import combinations

def _score(U):
    """Compute the exponent score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    M = max(U)
    if M == 0:
        return 0.0
    # Compute U+U and U-U using bitsets
    # U+U: sums range from 0 to 2M
    sumset = np.zeros(2*M + 1, dtype=bool)
    for a in U:
        sumset[a:a+n] |= np.array(U, dtype=bool)  # This is wrong, let me fix
    
    # Better: use numpy shifts
    ub = np.zeros(M + 1, dtype=bool)
    ub[U] = True
    
    # U+U
    sumset = np.zeros(2*M + 1, dtype=bool)
    for a in U:
        sumset[a:a+len(U)] |= ub  # This is still wrong
    
    # Correct approach:
    sumset = np.zeros(2*M + 1, dtype=bool)
    for a in U:
        shifted = np.zeros(2*M + 1, dtype=bool)
        shifted[a:a+M+1] = ub
        sumset |= shifted
    
    # U-U: differences range from -M to M
    diffset = np.zeros(2*M + 1, dtype=bool)  # index i represents value i-M
    for a in U:
        for b in U:
            d = a - b + M  # offset to make non-negative
            diffset[d] = True
    
    len_sum = sumset.sum()
    len_diff = diffset.sum()
    
    if len_sum == 0:
        return 0.0
    
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    exponent = 1.0 + math.log(ratio) / math.log(2*M + 1)
    return exponent


def _score_fast(U):
    """Faster score using numpy."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    ub = np.zeros(M + 1, dtype=bool)
    ub[U] = True
    
    # U+U via convolution
    # Use FFT for speed
    f = np.zeros(2*M + 1)
    f[U] = 1.0
    conv = np.fft.irfft(np.fft.rfft(f, n=2*M+1) ** 2, n=2*M+1)
    sumset_count = np.sum(conv > 0.5)
    
    # U-U: f * f_reversed
    f_rev = f[::-1]
    conv_diff = np.fft.irfft(np.fft.rfft(f, n=4*M+1) * np.fft.rfft(f_rev, n=4*M+1), n=4*M+1)
    # The difference a-b ranges from -M to M, centered at M in the convolution
    # Actually, let me think: conv_diff[k] = sum_i f[i]*f_rev[k-i] = sum_i f[i]*f[M-(k-i)] = sum_i f[i]*f[M+k-i]
    # This is the cross-correlation. Let me just do it directly for correctness.
    
    # Direct computation for U-U (should be fast enough for moderate M)
    diffset = np.zeros(2*M + 1, dtype=bool)
    U_arr = np.array(U)
    for a in U_arr:
        diffset[a:a+M+1] |= ub  # WRONG - this computes a + U not a - U
    
    # Correct: a - b for a,b in U. a-b ranges from -M to M.
    # (a-b) + M ranges from 0 to 2M
    # For fixed a: a - b for b in U gives values a - U
    # a - b + M = a + M - b
    # So diffset[a+M-b] = True for b in U
    # That is: diffset[a+M : a+M+M+1] shifted... 
    # diffset indices: for b in U, index = a + M - b
    # So diffset[a+M-b] = True, which means diffset[(a+M-b)] = True
    # Equivalently: diffset[a+M : a+M+M+1] |= ub reversed? No.
    # diffset[a+M-b] for b in U: the set of indices is {a+M-b : b in U}
    # = a+M - U
    # So diffset[a+M-U] = True
    # In numpy: diffset[a+M-b] for each b, or diffset[(a+M)-U_arr] = True
    
    for a in U:
        diffset[(a + M) - U_arr] = True
    
    len_sum = sumset_count
    len_diff = diffset.sum()
    
    if len_sum == 0:
        return 0.0
    
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    exponent = 1.0 + math.log(ratio) / math.log(2*M + 1)
    return exponent


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def consider(U):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return
        U = sorted(set(U))
        if 0 not in U or len(U) < 2:
            return
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Try seed
    consider([0, 1, 3])
    
    # Try various structured sets
    # 1. Small exhaustive search for sets with small max
    # For max M up to ~30, try all subsets containing 0
    # That's 2^M subsets, too many for M=30. Let's be smarter.
    
    # Try sets of form {0, a, b, c, ...} with small elements
    # Systematic search over small sets
    
    # Try all sets with max <= 15 (2^15 = 32768 subsets, manageable)
    for M in range(2, 16):
        if time.time() > deadline:
            break
        # Try subsets of {0, 1, ..., M} that contain 0 and M
        # But 2^(M-1) subsets... for M=15 that's 16384, OK
        # Actually let's just try specific patterns
        
        # Arithmetic progressions
        for step in range(1, M+1):
            U = list(range(0, M+1, step))
            if M in U:
                consider(U)
        
        # Two blocks: {0,...,a} ∪ {b,...,M}
        for a in range(1, M):
            for b in range(a+1, M+1):
                U = list(range(0, a+1)) + list(range(b, M+1))
                consider(U)
    
    # Try sets with specific structure: {0, a, 2a, ..., ka, ka+1, ka+2, ..., ka+b}
    for a in range(1, 20):
        for k in range(1, 20):
            for b in range(0, 10):
                M = a*k + b
                if M > 200:
                    break
                U = list(range(0, a*(k+1), a)) + list(range(a*k+1, a*k+b+1))
                consider(U)
    
    # Try geometric-like sets
    for base in range(2, 10):
        for k in range(2, 15):
            U = [base**i for i in range(k)]
            if max(U) < 5000:
                consider(U)
    
    # Try random sets
    import random
    random.seed(42)
    
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Random size and max
        n = random.randint(3, 20)
        M = random.randint(5, 200)
        
        # Generate random set containing 0
        U = [0]
        remaining = list(range(1, M+1))
        U.extend(random.sample(remaining, min(n-1, len(remaining))))
        U = sorted(set(U))
        
        if len(U) < 2:
            continue
        
        consider(U)
    
    # Local search: try to improve best_set by adding/removing elements
    if time.time() < deadline:
        current = list(best_set)
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            M = max(current)
            
            # Try adding a new element
            for new_elem in range(1, M + 50):
                if new_elem in current:
                    continue
                candidate = sorted(current + [new_elem])
                s = _score_fast(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    current = candidate
                    improved = True
                    break
            
            # Try removing an element (not 0)
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    candidate = current[:i] + current[i+1:]
                    if len(candidate) < 2:
                        continue
                    s = _score_fast(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = candidate
                        improved = True
                        break
    
    return best_set
