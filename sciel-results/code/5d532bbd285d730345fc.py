"""Improved program for two Sidon sets with disjoint difference sets."""
import random
import numpy as np

def _is_sidon(arr):
    """Check if arr is a Sidon set (all pair sums a+b with a<=b are unique)."""
    sums = set()
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _diffs(arr):
    """Return set of nonzero differences |b-a| for a != b in arr."""
    d = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(abs(arr[j] - arr[i]))
    return d

def construct():
    n = 256
    # We'll use a greedy randomized approach with multiple restarts
    # Strategy: alternate adding to A and B, preferring numbers that don't conflict
    
    best_total = 0
    best_A = []
    best_B = []
    
    for seed in range(50):
        rng = random.Random(seed)
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        # Precompute candidate pool
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        # Greedy: try to add each candidate to A or B
        for x in candidates:
            # Try adding to A
            A_new = A + [x]
            if _is_sidon(A_new):
                d_new = _diffs(A_new)
                if not (d_new & diffs_B):
                    A = A_new
                    diffs_A = d_new
                    continue
            
            # Try adding to B
            B_new = B + [x]
            if _is_sidon(B_new):
                d_new = _diffs(B_new)
                if not (d_new & diffs_A):
                    B = B_new
                    diffs_B = d_new
                    continue
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Also try a deterministic greedy from 1 to 256, alternating sides
    for side_first in (0, 1):
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in range(1, n + 1):
            # Decide which side to try first based on current sizes
            if side_first == 0:
                order = (0, 1) if len(A) <= len(B) else (1, 0)
            else:
                order = (1, 0) if len(B) <= len(A) else (0, 1)
            
            placed = False
            for side in order:
                if side == 0:
                    A_new = A + [x]
                    if _is_sidon(A_new):
                        d_new = _diffs(A_new)
                        if not (d_new & diffs_B):
                            A = A_new
                            diffs_A = d_new
                            placed = True
                            break
                else:
                    B_new = B + [x]
                    if _is_sidon(B_new):
                        d_new = _diffs(B_new)
                        if not (d_new & diffs_A):
                            B = B_new
                            diffs_B = d_new
                            placed = True
                            break
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    assert _is_sidon(best_A), "A is not Sidon"
    assert _is_sidon(best_B), "B is not Sidon"
    assert not (_diffs(best_A) & _diffs(best_B)), "Differences not disjoint"
    assert len(best_A) <= 64 and len(best_B) <= 64, "Size limit exceeded"
    assert all(1 <= x <= 256 for x in best_A + best_B), "Out of range"
    
    return [best_A, best_B]
