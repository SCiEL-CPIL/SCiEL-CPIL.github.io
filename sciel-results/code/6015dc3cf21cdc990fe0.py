"""Additive base with few representations: NumPy-accelerated local search with targeted moves.

Uses convolution for fast multiplicity computation and prioritizes moves that
reduce the peak multiplicity by targeting high-count sums.
"""
import time
import random
import numpy as np

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 80.0


def _compute_multiplicities_fast(A):
    """Compute unordered-pair representation counts for all s in [128,256] using convolution.
    
    Returns array of length 257 where mult[s] = number of unordered pairs (a,b) in A
    with a <= b and a+b = s.
    """
    if not A:
        return np.zeros(257, dtype=np.int32)
    
    A_arr = np.array(sorted(A), dtype=np.int32)
    # Convolution: indicator vector convolved with itself
    ind = np.zeros(129, dtype=np.int32)
    ind[A_arr] = 1
    conv = np.convolve(ind, ind, mode='full')  # length 257
    
    # conv[s] counts ordered pairs (a,b) with a+b=s, a,b in [0,128]
    # For unordered pairs with a<=b:
    # If s is even and s/2 in A: unordered_count = (conv[s] - 1) // 2 + 1 = (conv[s] + 1) // 2
    # If s is odd: unordered_count = conv[s] // 2
    # More generally: unordered = (conv[s] + (1 if s%2==0 and s//2 in A else 0)) // 2
    
    A_set = set(A)
    mult = np.zeros(257, dtype=np.int32)
    for s in range(128, 257):
        if s % 2 == 0 and s // 2 in A_set:
            mult[s] = (conv[s] + 1) // 2
        else:
            mult[s] = conv[s] // 2
    
    return mult


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    mult = _compute_multiplicities_fast(A)
    return int(np.max(mult[128:257]))


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    mult = _compute_multiplicities_fast(A)
    return bool(np.all(mult[128:257] > 0))


def _get_high_mult_elements(A, mult, threshold):
    """Find elements that appear in sums with multiplicity >= threshold."""
    if not A:
        return set()
    
    A_set = set(A)
    high_elements = set()
    for s in range(128, 257):
        if mult[s] >= threshold:
            for a in A:
                b = s - a
                if a <= b and b in A_set:
                    high_elements.add(a)
                    high_elements.add(b)
    return high_elements


def _generate_random_valid_set(min_size=30, max_size=80):
    """Generate a random set that is likely to cover [128,256]."""
    # Start with a guaranteed valid base
    base = set(range(64, 129))
    
    # Randomly remove some elements while maintaining validity
    A = set(base)
    while len(A) > min_size and time.time() < _DEADLINE - 5:
        candidates = list(A)
        random.shuffle(candidates)
        removed = False
        for elem in candidates:
            new_A = A - {elem}
            if _is_valid(new_A):
                A = new_A
                removed = True
                break
        if not removed:
            break
    
    # Optionally add some random elements to explore different structures
    if len(A) < max_size:
        for v in range(0, 129):
            if v not in A and random.random() < 0.3:
                A.add(v)
                if len(A) >= max_size:
                    break
    
    return A


def _local_search(A, max_iterations=10000, time_budget=None):
    """Targeted local search: prioritize moves affecting high-multiplicity sums."""
    if time_budget is None:
        time_budget = _DEADLINE - time.time() - 5
    
    A = list(set(A))
    mult = _compute_multiplicities_fast(A)
    peak = int(np.max(mult[128:257]))
    best_A = A[:]
    best_peak = peak
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Identify high-multiplicity elements for targeted removal
        if peak > 1:
            high_elems = _get_high_mult_elements(A, mult, peak)
            if high_elems:
                # Try removing a high-multiplicity element
                candidates = [e for e in A if e in high_elems]
                random.shuffle(candidates)
                for elem in candidates[:3]:  # Try up to 3 candidates
                    cand = [x for x in A if x != elem]
                    if _is_valid(cand):
                        p = _peak(cand)
                        if p < peak:
                            A = cand
                            mult = _compute_multiplicities_fast(A)
                            peak = p
                            best_A = A[:]
                            best_peak = peak
                            break
                        elif p == peak and random.random() < 0.2:
                            A = cand
                            mult = _compute_multiplicities_fast(A)
                            break
        
        # Try removing a random element (for diversity)
        if len(A) > 10:
            idx = random.randint(0, len(A) - 1)
            cand = A[:idx] + A[idx + 1:]
            if _is_valid(cand):
                p = _peak(cand)
                if p < peak:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < 0.1:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
        
        # Try adding a random element
        if len(A) < 128:
            v = random.randint(0, 128)
            if v not in A:
                cand = A + [v]
                p = _peak(cand)
                if p < peak:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < 0.05:
                    A = cand
                    mult = _compute_multiplicities_fast(A)
        
        # Try swapping: remove one, add another
        if len(A) > 5 and len(A) < 127:
            idx = random.randint(0, len(A) - 1)
            v = random.randint(0, 128)
            if v not in A:
                cand = A[:idx] + A[idx + 1:] + [v]
                if _is_valid(cand):
                    p = _peak(cand)
                    if p < peak:
                        A = cand
                        mult = _compute_multiplicities_fast(A)
                        peak = p
                        best_A = A[:]
                        best_peak = peak
                    elif p == peak and random.random() < 0.1:
                        A = cand
                        mult = _compute_multiplicities_fast(A)
    
    return best_A, best_peak


def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate a random valid starting set
        A = _generate_random_valid_set(min_size=25, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search
        A_opt, peak = _local_search(A, max_iterations=8000)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Early exit if we found optimal
        if best_peak <= 1:
            break
    
    # If no valid set found, use fallback
    if best_A is None:
        best_A = list(range(64, 129))
        best_peak = _peak(best_A)
    
    return sorted(best_A)
