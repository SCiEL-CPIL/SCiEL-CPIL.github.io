# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _pair_sums_unique(S):
    """Check if all pair sums (with replacement) in S are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diff_set(S):
    """Return set of nonzero absolute differences in S."""
    diffs = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(S[i] - S[j])
            if d > 0:
                diffs.add(d)
    return diffs

def _try_add_fast(S, x, used_sums, used_diffs, other_diffs):
    """Try to add x to Sidon set S. Returns (new_S, new_sums, new_diffs) or (None, None, None).
    S must be sorted. used_sums and used_diffs are the current sum/diff sets for S.
    other_diffs is the diff set of the OTHER Sidon set (must be disjoint)."""
    n = len(S)
    # Check Sidon property: all new sums x+S[i] must be unique and not in used_sums
    new_sums = set()
    for i in range(n):
        s = x + S[i]
        if s in used_sums or s in new_sums:
            return None, None, None
        new_sums.add(s)
    # Also x+x
    s_xx = x + x
    if s_xx in used_sums or s_xx in new_sums:
        return None, None, None
    new_sums.add(s_xx)
    
    # Check differences: x-S[i] must not conflict with other_diffs or used_diffs
    new_diffs = set()
    for i in range(n):
        d = abs(x - S[i])
        if d > 0:
            if d in other_diffs or d in used_diffs:
                return None, None, None
            new_diffs.add(d)
    
    return S + [x], used_sums | new_sums, used_diffs | new_diffs

def _greedy_interleave(universe, deadline):
    """Build A and B interleaved from a shuffled universe."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in universe:
        if time.time() > deadline:
            break
        # Try adding to A
        new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
        if new_A is not None:
            A = new_A
            sums_A = new_sums_A
            diffs_A = new_diffs_A
            continue
        # Try adding to B
        new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
        if new_B is not None:
            B = new_B
            sums_B = new_sums_B
            diffs_B = new_diffs_B
    
    return A, B

def _greedy_balanced(universe, deadline):
    """Build A and B in alternating rounds, trying to keep sizes balanced."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in universe:
        if time.time() > deadline:
            break
        # Prefer adding to the smaller set
        if len(A) <= len(B):
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A = new_diffs_A
                continue
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B = new_diffs_B
        else:
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B = new_diffs_B
                continue
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A = new_diffs_A
    
    return A, B

def _greedy_priority(universe, deadline):
    """Build A and B, preferring elements that create fewer conflicts."""
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    for x in universe:
        if time.time() > deadline:
            break
        # Score each potential addition by how many future options it blocks
        # Simple heuristic: try A first if it's smaller, else B
        if len(A) < len(B):
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A = new_diffs_A
                continue
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B = new_diffs_B
        elif len(B) < len(A):
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B = new_diffs_B
                continue
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A = new_diffs_A
        else:
            # Equal sizes, try both
            new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
            if new_A is not None:
                A = new_A
                sums_A = new_sums_A
                diffs_A = new_diffs_A
                continue
            new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
            if new_B is not None:
                B = new_B
                sums_B = new_sums_B
                diffs_B = new_diffs_B
    
    return A, B

def _local_search(A, B, sums_A, sums_B, diffs_A, diffs_B, deadline):
    """Try to improve by swapping elements or adding new ones."""
    best_total = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    best_sums_A, best_sums_B = set(sums_A), set(sums_B)
    best_diffs_A, best_diffs_B = set(diffs_A), set(diffs_B)
    
    universe = list(range(1, 257))
    
    # Try adding elements from universe to either set
    for x in universe:
        if time.time() > deadline:
            break
        if x in A or x in B:
            continue
        # Try adding to A
        new_A, new_sums_A, new_diffs_A = _try_add_fast(A, x, sums_A, diffs_A, diffs_B)
        if new_A is not None:
            total = len(new_A) + len(B)
            if total > best_total:
                best_total = total
                best_A, best_B = new_A, list(B)
                best_sums_A, best_sums_B = new_sums_A, set(sums_B)
                best_diffs_A, best_diffs_B = new_diffs_A, set(diffs_B)
                A, sums_A, diffs_A = new_A, new_sums_A, new_diffs_A
            continue
        # Try adding to B
        new_B, new_sums_B, new_diffs_B = _try_add_fast(B, x, sums_B, diffs_B, diffs_A)
        if new_B is not None:
            total = len(A) + len(new_B)
            if total > best_total:
                best_total = total
                best_A, best_B = list(A), new_B
                best_sums_A, best_sums_B = set(sums_A), new_sums_B
                best_diffs_A, best_diffs_B = set(diffs_A), new_diffs_B
                B, sums_B, diffs_B = new_B, new_sums_B, new_diffs_B
    
    return best_A, best_B, best_sums_A, best_sums_B, best_diffs_A, best_diffs_B

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
    best_total = 0
    best_A = []
    best_B = []
    
    # Multiple random restarts with different strategies
    strategies = [_greedy_interleave, _greedy_balanced, _greedy_priority]
    
    for trial in range(300):
        if time.time() > deadline:
            break
        
        # Create random ordering
        universe = list(range(1, 257))
        random.shuffle(universe)
        
        # Cycle through strategies
        strategy = strategies[trial % len(strategies)]
        A, B = strategy(universe, deadline)
        total = len(A) + len(B)
        
        if total > best_total:
            best_total = total
            best_A = A
            best_B = B
    
    # Also try sequential (original) approach as baseline
    A_orig = []
    B_orig = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Try A
        new_A, new_sums_A, new_diffs_A = _try_add_fast(A_orig, x, sums_A, diffs_A, diffs_B)
        if new_A is not None:
            A_orig = new_A
            sums_A = new_sums_A
            diffs_A = new_diffs_A
            continue
        # Try B
        new_B, new_sums_B, new_diffs_B = _try_add_fast(B_orig, x, sums_B, diffs_B, diffs_A)
        if new_B is not None:
            B_orig = new_B
            sums_B = new_sums_B
            diffs_B = new_diffs_B
    
    total_orig = len(A_orig) + len(B_orig)
    if total_orig > best_total:
        best_total = total_orig
        best_A = A_orig
        best_B = B_orig
    
    # Local search on the best solution found
    if best_A and best_B:
        # Recompute sums and diffs for local search
        sums_A_ls = set()
        for i in range(len(best_A)):
            for j in range(i, len(best_A)):
                sums_A_ls.add(best_A[i] + best_A[j])
        sums_B_ls = set()
        for i in range(len(best_B)):
            for j in range(i, len(best_B)):
                sums_B_ls.add(best_B[i] + best_B[j])
        diffs_A_ls = _diff_set(best_A)
        diffs_B_ls = _diff_set(best_B)
        
        ls_A, ls_B, _, _, _, _ = _local_search(best_A, best_B, sums_A_ls, sums_B_ls, diffs_A_ls, diffs_B_ls, deadline)
        total_ls = len(ls_A) + len(ls_B)
        if total_ls > best_total:
            best_total = total_ls
            best_A = ls_A
            best_B = ls_B
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
