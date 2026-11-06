"""Additive base with few representations: NumPy-accelerated local search with CSP-guided moves and sparse initialization.

Uses fast convolution-based multiplicity computation and a Most-Constrained-Variable
heuristic. Includes a sparse completion initializer to find low-multiplicity structures
more effectively than dense removal strategies.
"""
import time
import random
import numpy as np

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 70.0


def _compute_multiplicities_fast(A):
    """Compute unordered-pair representation counts for s in [128,256] using NumPy.
    
    Returns a numpy array of length 129 where index i gives the count for s=128+i.
    """
    A_arr = np.array(sorted(A), dtype=np.int64)
    A_set = set(A)
    n = len(A_arr)
    
    # Compute all pair sums (a_i + a_j) for i <= j
    # For efficiency, compute the full convolution then adjust for unordered pairs
    # indicator vector
    ind = np.zeros(257, dtype=np.int64)
    ind[A_arr] = 1
    
    # Full convolution gives ordered pairs (counting (a,b) and (b,a) separately)
    conv = np.convolve(ind, ind, mode='full')
    
    # For unordered pairs with equal summands allowed:
    # count_unordered(s) = (conv[s] + diag_count(s)) // 2
    # where diag_count(s) = 1 if s is even and s/2 in A, else 0
    # Actually: conv[s] counts ordered pairs (a,b) with a+b=s, a,b in A
    # Unordered pairs: pairs with a < b counted once, pairs with a == b counted once
    # conv[s] = 2 * (# pairs with a < b) + (# pairs with a == b)
    # So unordered count = # pairs with a < b + # pairs with a == b
    #                  = (conv[s] - diag) + diag = conv[s] - diag + diag... 
    # Let me re-derive: 
    # If a != b: contributes 2 to conv (as (a,b) and (b,a)), contributes 1 to unordered
    # If a == b: contributes 1 to conv, contributes 1 to unordered
    # Let d = 1 if s even and s/2 in A, else 0
    # conv[s] = 2*(#distinct unordered pairs with a!=b) + d
    # unordered_count = (#distinct unordered pairs with a!=b) + d
    # So unordered_count = (conv[s] - d) / 2 + d = (conv[s] + d) / 2
    
    result = np.zeros(129, dtype=np.int64)
    for i in range(129):
        s = 128 + i
        d = 1 if (s % 2 == 0 and (s // 2) in A_set) else 0
        result[i] = (conv[s] + d) // 2
    
    return result


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    mult = _compute_multiplicities_fast(A)
    return int(np.max(mult)) if len(mult) > 0 else 0


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    mult = _compute_multiplicities_fast(A)
    return bool(np.all(mult > 0))


def _element_frequency_in_high_mult(A, mult, threshold=None):
    """For each element in A, count how many high-multiplicity sums it participates in.
    
    Uses the Most Constrained Variable heuristic: elements that appear in many
    high-multiplicity sums are the best candidates for removal (to break those sums).
    
    Returns a dict mapping element -> frequency count.
    """
    if threshold is None:
        peak = int(np.max(mult))
        threshold = max(2, peak)  # Focus on sums at or near peak
    
    A_set = set(A)
    freq = {a: 0 for a in A}
    
    for i in range(129):
        s = 128 + i
        if mult[i] >= threshold:
            for a in A:
                b = s - a
                if b >= a and b in A_set:
                    freq[a] += 1
                    if b != a:
                        freq[b] += 1
    
    return freq


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


def _generate_sparse_valid_set():
    """Generate a sparse valid set by completing a random seed set."""
    # Start with a small random set
    A = set(random.sample(range(0, 129), 10))
    
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Check for uncovered sums
        A_list = sorted(A)
        A_set = set(A)
        mult = _compute_multiplicities_fast(A)
        
        # Find first uncovered sum
        uncovered_idx = -1
        for i in range(129):
            if mult[i] == 0:
                uncovered_idx = i
                break
        
        if uncovered_idx == -1:
            # All covered
            return A
        
        s = 128 + uncovered_idx
        
        # Try to add an element to cover s
        # We need a in A such that b = s - a is in A.
        # Currently no such pair exists.
        # We can add a new element x.
        # If we add x, we need x + y = s for some y in A, OR y + x = s for some y in A, OR x + x = s.
        # So we need to find x such that (s - x) in A OR (s - x) == x (if 2x=s).
        
        candidates = []
        for a in A:
            b = s - a
            if 0 <= b <= 128:
                # If we add b, then a+b=s is covered.
                # b might already be in A? No, if b was in A, mult would be > 0.
                # So b is not in A.
                candidates.append(b)
        
        # Also consider x = s/2
        if s % 2 == 0:
            x = s // 2
            if 0 <= x <= 128 and x not in A:
                candidates.append(x)
        
        if not candidates:
            # Cannot cover s by adding a single element to current A?
            # This means for all a in A, s-a is out of bounds [0, 128].
            # We might need to add multiple elements or restart.
            return None
            
        # Choose candidate that minimizes increase in peak multiplicity
        best_cand = None
        best_peak = float('inf')
        
        # Shuffle for tie-breaking
        random.shuffle(candidates)
        
        for cand in candidates[:5]: # Limit search
            new_A = A | {cand}
            new_mult = _compute_multiplicities_fast(new_A)
            p = int(np.max(new_mult))
            if p < best_peak:
                best_peak = p
                best_cand = cand
        
        if best_cand is not None:
            A.add(best_cand)
        else:
            break
            
    # Verify validity
    if _is_valid(A):
        return A
    return None


def _local_search(A, max_iterations=5000, time_budget=None):
    """CSP-guided local search: prioritize elements in high-multiplicity sums.
    
    Uses the Most Constrained Variable heuristic: elements that appear in many
    high-multiplicity sums are prioritized for removal or swap, as removing them
    is most likely to reduce the peak multiplicity.
    """
    if time_budget is None:
        time_budget = _DEADLINE - time.time() - 5
    
    A = list(set(A))
    mult = _compute_multiplicities_fast(A)
    peak = int(np.max(mult))
    best_A = A[:]
    best_peak = peak
    
    # Cache for frequency computation
    freq_cache = None
    freq_cache_peak = -1
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Recompute frequency if peak changed or periodically
        if freq_cache_peak != peak or iteration % 50 == 0:
            freq_cache = _element_frequency_in_high_mult(A, mult, threshold=peak)
            freq_cache_peak = peak
        
        # Sort elements by frequency (descending) for prioritized selection
        if freq_cache:
            sorted_elements = sorted(A, key=lambda x: freq_cache.get(x, 0), reverse=True)
        else:
            sorted_elements = A[:]
            random.shuffle(sorted_elements)
        
        # With 70% probability, pick from top 30% (high-frequency elements)
        # With 30% probability, pick randomly (for diversity)
        if random.random() < 0.7 and len(sorted_elements) > 3:
            top_n = max(1, len(sorted_elements) // 3)
            candidate_pool = sorted_elements[:top_n]
        else:
            candidate_pool = sorted_elements
        
        # Try removing a high-frequency element
        if len(A) > 10:
            elem = random.choice(candidate_pool)
            if elem in A:
                idx = A.index(elem)
                cand = A[:idx] + A[idx + 1:]
                if _is_valid(cand):
                    new_mult = _compute_multiplicities_fast(cand)
                    p = int(np.max(new_mult))
                    if p < peak:
                        A = cand
                        peak = p
                        mult = new_mult
                        best_A = A[:]
                        best_peak = peak
                        freq_cache = None  # Invalidate cache
                    elif p == peak and random.random() < 0.15:
                        A = cand
                        mult = new_mult
        
        # Try adding a random element
        if len(A) < 128:
            v = random.randint(0, 128)
            if v not in A:
                cand = A + [v]
                new_mult = _compute_multiplicities_fast(cand)
                p = int(np.max(new_mult))
                if p < peak:
                    A = cand
                    peak = p
                    mult = new_mult
                    best_A = A[:]
                    best_peak = peak
                    freq_cache = None
                elif p == peak and random.random() < 0.05:
                    A = cand
                    mult = new_mult
        
        # Try swapping: remove a high-frequency element, add another
        if len(A) > 5 and len(A) < 127:
            elem = random.choice(candidate_pool)
            if elem in A:
                idx = A.index(elem)
                v = random.randint(0, 128)
                if v not in A:
                    cand = A[:idx] + A[idx + 1:] + [v]
                    if _is_valid(cand):
                        new_mult = _compute_multiplicities_fast(cand)
                        p = int(np.max(new_mult))
                        if p < peak:
                            A = cand
                            peak = p
                            mult = new_mult
                            best_A = A[:]
                            best_peak = peak
                            freq_cache = None
                        elif p == peak and random.random() < 0.1:
                            A = cand
                            mult = new_mult
    
    return best_A, best_peak


def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts with varied parameters
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Alternate between dense removal and sparse completion strategies
        if restart % 2 == 0:
            # Dense removal strategy
            min_size = 25 + (restart % 3) * 5
            max_size = 60 + (restart % 4) * 10
            A = _generate_random_valid_set(min_size=min_size, max_size=max_size)
        else:
            # Sparse completion strategy
            A = _generate_sparse_valid_set()
            if A is None:
                # Fallback to dense if sparse fails
                A = _generate_random_valid_set(min_size=30, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search with more iterations for early restarts
        remaining = _DEADLINE - time.time() - 10
        if remaining > 20:
            max_iter = 5000
        elif remaining > 10:
            max_iter = 3000
        else:
            max_iter = 1500
            
        A_opt, peak = _local_search(A, max_iterations=max_iter)
        
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
