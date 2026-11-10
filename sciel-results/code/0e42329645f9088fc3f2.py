"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_mult(A, n=128):
    """Compute representation multiplicities for sums in [n, 2n].
    Unordered pairs with equal summands allowed."""
    A_set = set(A)
    # multiplicities for sums s in [n, 2n]
    mult = np.zeros(2*n + 1, dtype=np.int32)
    for a in A:
        for b in A:
            if a <= b:  # unordered, count each pair once
                s = a + b
                if n <= s <= 2*n:
                    mult[s] += 1
    return mult

def _compute_mult_fast(A, n=128):
    """Faster multiplicity computation using convolution-like approach.
    For unordered pairs with equal summands: 
    count(s) = number of (a,b) in A x A with a<=b and a+b=s
    = (number of ordered pairs (a,b) in A x A with a+b=s) 
      adjusted for self-pairs
    Actually for unordered with equal allowed:
    If a != b: pair {a,b} counted once
    If a == b: pair {a,a} counted once
    
    Ordered count: C(s) = |{(a,b) in A^2 : a+b=s}|
    Unordered count: U(s) = (C(s) + D(s)) / 2 where D(s) = |{a in A : 2a=s}|
    Because each unordered pair {a,b} with a!=b corresponds to 2 ordered pairs,
    and each {a,a} corresponds to 1 ordered pair.
    So C(s) = 2 * (# unordered with a!=b) + 1 * (# unordered with a==b)
    U(s) = (# unordered with a!=b) + (# unordered with a==b)
         = (C(s) - D(s))/2 + D(s) = (C(s) + D(s))/2
    """
    A_arr = np.array(sorted(A), dtype=np.int64)
    # Convolution to get ordered counts
    # Use FFT for speed
    size = 2 * n + 1
    # Pad to power of 2 for FFT
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    f = np.zeros(fft_size, dtype=np.float64)
    for a in A_arr:
        f[a] = 1.0
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=fft_size)
    conv = np.round(conv).astype(np.int64)
    
    # D(s) = 1 if s is even and s//2 in A
    A_set = set(A)
    D = np.zeros(2*n + 1, dtype=np.int64)
    for s in range(n, 2*n + 1):
        if s % 2 == 0 and s // 2 in A_set:
            D[s] = 1
    
    mult = np.zeros(2*n + 1, dtype=np.int64)
    for s in range(n, 2*n + 1):
        mult[s] = (conv[s] + D[s]) // 2
    
    return mult

def _peak_mult(mult, n=128):
    return int(np.max(mult[n:2*n+1]))

def _is_valid(A, n=128):
    """Check every integer in [n, 2n] has at least one representation."""
    mult = _compute_mult_fast(A, n)
    return np.all(mult[n:2*n+1] >= 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    n = 128
    
    # Start with a good initial solution
    # The seed uses [64, 128] which gives peak multiplicity 128
    # Let's try a better starting point
    # A random subset of [0,128] with enough elements
    
    random.seed(42)
    
    # Strategy: use a structured set that covers [128,256]
    # Start with a moderately dense random set
    best_A = None
    best_peak = float('inf')
    
    # Try several random initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random subset of [0, 128] with size around 40-60
        size = random.randint(35, 55)
        A = set(random.sample(range(0, 129), size))
        
        # Ensure coverage: check if [128,256] is covered
        # If not, add elements to fix gaps
        mult = _compute_mult_fast(A, n)
        gaps = [s for s in range(n, 2*n+1) if mult[s] == 0]
        
        if gaps:
            # Try to fix by adding elements
            changed = True
            max_fixes = 100
            while gaps and max_fixes > 0:
                max_fixes -= 1
                # Find an element that covers the most gaps
                best_elem = None
                best_cover = 0
                for cand in range(0, 129):
                    if cand in A:
                        continue
                    # How many gaps can cand cover?
                    # cand + x = s for some x in A, s in gaps
                    cover = 0
                    for g in gaps:
                        other = g - cand
                        if 0 <= other <= 128 and other in A:
                            cover += 1
                    if cover > best_cover:
                        best_cover = cover
                        best_elem = cand
                if best_elem is not None and best_cover > 0:
                    A.add(best_elem)
                    mult = _compute_mult_fast(A, n)
                    gaps = [s for s in range(n, 2*n+1) if mult[s] == 0]
                else:
                    break
        
        if not gaps:
            peak = _peak_mult(mult, n)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    # Now do local search to improve
    A = best_A
    mult = _compute_mult_fast(A, n)
    peak = _peak_mult(mult, n)
    
    # Local search: try swaps (remove one, add one) and single additions/removals
    # Focus on reducing peak multiplicity
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        no_improve += 1
        
        if no_improve > max_no_improve:
            # Try a random restart of local search parameters
            no_improve = 0
            max_no_improve += 20
        
        # Try all possible swaps
        A_list = sorted(A)
        tried = set()
        
        for a_remove in A_list:
            if time.time() > deadline:
                break
            
            for a_add in range(0, 129):
                if a_add in A or a_add == a_remove:
                    continue
                
                key = (a_remove, a_add)
                if key in tried:
                    continue
                tried.add(key)
                
                # Compute new multiplicity efficiently
                # Remove a_remove, add a_add
                # Delta for removing a_remove:
                # For each b in A (b != a_remove, b < a_remove for unordered):
                #   sum s = a_remove + b loses 1
                # For b = a_remove: sum 2*a_remove loses 1
                # Delta for adding a_add:
                # For each b in A (b != a_add, b > a_add for unordered to avoid double count):
                #   Actually, let's just recompute for simplicity if A is small enough
                pass
        
        # Simpler approach: just recompute for promising swaps
        # Try a limited number of random swaps per iteration
        n_swaps = 200
        for _ in range(n_swaps):
            if time.time() > deadline:
                break
            
            a_remove = random.choice(list(A))
            a_add = random.randint(0, 128)
            
            if a_add in A:
                continue
            
            new_A = A - {a_remove} | {a_add}
            new_mult = _compute_mult_fast(new_A, n)
            
            if not np.all(new_mult[n:2*n+1] >= 1):
                continue
            
            new_peak = _peak_mult(new_mult, n)
            
            if new_peak < peak:
                A = new_A
                mult = new_mult
                peak = new_peak
                improved = True
                no_improve = 0
                max_no_improve = 50
            elif new_peak == peak:
                # Accept with some probability for exploration
                if random.random() < 0.1:
                    A = new_A
                    mult = new_mult
        
        if not improved:
            # Try single removals (if still valid)
            for a_remove in list(A):
                if time.time() > deadline:
                    break
                new_A = A - {a_remove}
                if len(new_A) < 10:
                    continue
                new_mult = _compute_mult_fast(new_A, n)
                if np.all(new_mult[n:2*n+1] >= 1):
                    new_peak = _peak_mult(new_mult, n)
                    if new_peak < peak:
                        A = new_A
                        mult = new_mult
                        peak = new_peak
                        improved = True
                        no_improve = 0
        
        if not improved:
            # Try single additions
            for a_add in range(0, 129):
                if time.time() > deadline:
                    break
                if a_add in A:
                    continue
                new_A = A | {a_add}
                new_mult = _compute_mult_fast(new_A, n)
                new_peak = _peak_mult(new_mult, n)
                if new_peak < peak:
                    A = new_A
                    mult = new_mult
                    peak = new_peak
                    improved = True
                    no_improve = 0
    
    return sorted(A)

# EVOLVE-BLOCK-END
