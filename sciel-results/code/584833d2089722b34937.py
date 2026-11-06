"""Additive base with few representations: constructive mixed-radix basis + local search.

Constructs a set A subset [0,128] such that every integer in [128,256] is a sum of
two elements of A (unordered pairs, equal summands allowed), minimizing the peak
representation multiplicity.

Key construction: Use a two-tier structure where A = L ∪ H with L = {0,...,w-1}
and H = {w + k*d : k=0,...,m-1} for suitable parameters, then verify and polish
with local search.
"""
import time
import random
import numpy as np

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 70.0


def _compute_multiplicities(A):
    """Compute representation counts for all s in [128,256]."""
    A_set = set(A)
    A_sorted = sorted(A)
    mult = np.zeros(129, dtype=np.int32)  # index 0 -> s=128, index 128 -> s=256
    for s in range(128, 257):
        count = 0
        for a in A_sorted:
            if a > s:
                break
            b = s - a
            if b >= a and b in A_set:
                count += 1
        mult[s - 128] = count
    return mult


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    mult = _compute_multiplicities(A)
    return int(mult.max())


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    mult = _compute_multiplicities(A)
    return bool(np.all(mult > 0))


def _construct_radix_basis():
    """
    Construct a mixed-radix basis aiming for peak=1.
    
    Strategy: We need |A|(|A|+1)/2 >= 129, so |A| >= 16.
    
    Use a structured construction:
    - Let w be a "width" parameter.
    - L = {0, 1, ..., w-1}
    - H = {w + k * stride : k = 0, ..., m-1} where stride is chosen to avoid collisions.
    
    For unique representation, we need:
    1. No two distinct pairs (a_i, a_j) give the same sum.
    2. All sums in [128,256] are covered.
    
    A simpler approach: use a greedy construction.
    Start with an empty set and add elements one by one, each time choosing
    the element that covers the most uncovered sums while minimizing collisions.
    """
    # Greedy construction: build A to cover [128,256] with minimal peak
    target_sums = set(range(128, 257))
    A = set()
    
    # Precompute: for each candidate element x in [0,128], which sums does it help with?
    # When we add x to A, it forms pairs with all existing elements y in A:
    #   x + y for y in A (and x + x)
    
    # Start by ensuring we have enough elements
    # Try different starting configurations
    
    best_A = None
    best_peak = float('inf')
    
    # Try several parameterized constructions
    for w in [8, 12, 16, 20, 24, 32]:
        for stride in [1, 2, 3, 4, 5, 6, 8]:
            L = set(range(0, w))
            # H elements: w + k*stride, must be <= 128
            H = set()
            k = 0
            while w + k * stride <= 128:
                H.add(w + k * stride)
                k += 1
            
            A_candidate = L | H
            
            if not A_candidate:
                continue
            
            if not _is_valid(A_candidate):
                continue
            
            p = _peak(A_candidate)
            if p < best_peak:
                best_peak = p
                best_A = set(A_candidate)
            
            if best_peak == 1:
                return sorted(best_A)
    
    # If no simple construction works, use greedy
    if best_A is None:
        # Greedy: start with a base and add elements
        A = set(range(0, 64))
        # Ensure validity
        if not _is_valid(A):
            A = set(range(0, 129))
        
        # Try to reduce peak by removing elements
        changed = True
        while changed:
            changed = False
            for elem in sorted(A, reverse=True):
                if elem == 0:
                    continue
                new_A = A - {elem}
                if _is_valid(new_A):
                    old_peak = _peak(A)
                    new_peak = _peak(new_A)
                    if new_peak <= old_peak:
                        A = new_A
                        changed = True
        best_A = A
        best_peak = _peak(A)
    
    return sorted(best_A)


def _local_search(A, max_iterations=3000):
    """Local search: try removals, additions, and swaps to minimize peak."""
    A = list(set(A))
    peak = _peak(A)
    best_A = A[:]
    best_peak = peak
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        improved = False
        
        # Try removing a random element
        if len(A) > 12:
            idx = random.randint(0, len(A) - 1)
            cand = A[:idx] + A[idx + 1:]
            if _is_valid(cand):
                p = _peak(cand)
                if p < peak:
                    A = cand
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                    improved = True
                elif p == peak and random.random() < 0.1:
                    A = cand
        
        # Try adding a random element
        if len(A) < 128:
            v = random.randint(0, 128)
            if v not in A:
                cand = A + [v]
                p = _peak(cand)
                if p < peak:
                    A = cand
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                    improved = True
                elif p == peak and random.random() < 0.05:
                    A = cand
        
        # Try swapping: remove one, add another
        if len(A) > 8 and len(A) < 127:
            idx = random.randint(0, len(A) - 1)
            v = random.randint(0, 128)
            if v not in A:
                cand = A[:idx] + A[idx + 1:] + [v]
                if _is_valid(cand):
                    p = _peak(cand)
                    if p < peak:
                        A = cand
                        peak = p
                        best_A = A[:]
                        best_peak = peak
                        improved = True
                    elif p == peak and random.random() < 0.1:
                        A = cand
        
        if best_peak == 1:
            break
    
    return best_A, best_peak


def _greedy_construct():
    """
    Greedy construction: build A element by element to cover [128,256] with low peak.
    
    At each step, try adding each candidate element and pick the one that
    minimizes the resulting peak (or maximizes coverage if peak is tied).
    """
    A = set()
    
    # We need at least 16 elements (16*17/2 = 136 >= 129)
    min_size = 16
    
    for step in range(40):  # try to build up to 40 elements
        if time.time() > _DEADLINE - 10:
            break
        
        best_candidate = None
        best_peak = float('inf')
        best_covered = 0
        
        for x in range(0, 129):
            if x in A:
                continue
            cand_A = A | {x}
            
            if not _is_valid(cand_A):
                continue
            
            p = _peak(cand_A)
            mult = _compute_multiplicities(cand_A)
            covered = int(np.sum(mult > 0))
            
            if p < best_peak or (p == best_peak and covered > best_covered):
                best_peak = p
                best_covered = covered
                best_candidate = x
        
        if best_candidate is not None:
            A.add(best_candidate)
            if best_peak == 1 and len(A) >= min_size:
                break
        else:
            break
    
    return sorted(A)


def construct():
    t0 = time.time()
    random.seed(42)
    
    best_A = None
    best_peak = float('inf')
    
    # Method 1: Parameterized construction
    A0 = _construct_radix_basis()
    if A0 and _is_valid(A0):
        peak0 = _peak(A0)
        best_A = A0
        best_peak = peak0
    
    if best_peak == 1:
        return sorted(best_A)
    
    # Method 2: Greedy construction
    if time.time() < _DEADLINE - 15:
        A1 = _greedy_construct()
        if A1 and _is_valid(A1):
            peak1 = _peak(A1)
            if peak1 < best_peak:
                best_peak = peak1
                best_A = A1
    
    if best_peak == 1:
        return sorted(best_A)
    
    # Method 3: Local search from best so far
    if time.time() < _DEADLINE - 10 and best_A is not None:
        A_opt, peak = _local_search(best_A, max_iterations=3000)
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
    
    if best_peak == 1:
        return sorted(best_A)
    
    # Method 4: Random restarts
    for restart in range(8):
        if time.time() > _DEADLINE - 10:
            break
        
        # Random valid starting set
        base_size = random.randint(16, 40)
        A = set(random.sample(range(0, 129), base_size))
        
        # Ensure validity by adding elements if needed
        if not _is_valid(A):
            # Try to fix by adding elements
            for _ in range(50):
                if _is_valid(A):
                    break
                missing = [s for s in range(128, 257) 
                          if not any(a + b == s for a in A for b in A if a <= b)]
                if not missing:
                    break
                # Find elements that would cover missing sums
                candidates = set()
                for s in missing:
                    for a in range(0, 129):
                        b = s - a
                        if 0 <= b <= 128 and a <= b:
                            candidates.add(a)
                            candidates.add(b)
                if candidates:
                    v = random.choice(sorted(candidates))
                    if v not in A:
                        A.add(v)
        
        if not _is_valid(A):
            continue
        
        A_opt, peak = _local_search(sorted(A), max_iterations=1500)
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        if best_peak == 1:
            break
    
    # Fallback: ensure we return something valid
    if best_A is None or not _is_valid(best_A):
        best_A = list(range(0, 129))
        best_peak = _peak(best_A)
    
    return sorted(best_A)
