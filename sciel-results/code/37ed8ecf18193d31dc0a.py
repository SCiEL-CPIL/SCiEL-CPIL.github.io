"""Improved program for two Sidon sets with disjoint difference sets."""
import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before 90s kill


def _pair_sums(s):
    """Return set of all a+b for a,b in s with a<=b (i.e., combinations_with_replacement)."""
    n = len(s)
    result = set()
    for i in range(n):
        a = s[i]
        result.add(2 * a)
        for j in range(i + 1, n):
            result.add(a + s[j])
    return result


def _diffs(s):
    """Return set of all nonzero differences b-a for a,b in s, a!=b."""
    n = len(s)
    result = set()
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            result.add(s[j] - a)
    return result


def _valid_sidon(s):
    """Check if s is a Sidon set (all pair sums a+b with a<=b are unique)."""
    sums = _pair_sums(s)
    return len(sums) == len(s) * (len(s) + 1) // 2


def _try_add(A, B, x, diffA, diffB, sumA, sumB):
    """
    Try to add x to A or B. Returns (side, new_diffs, new_sums) if successful.
    side is 0 for A, 1 for B.
    """
    # Try adding to A
    new_diffs_A = set()
    for a in A:
        d = x - a
        if d != 0:
            new_diffs_A.add(d)
        d = a - x
        if d != 0:
            new_diffs_A.add(d)
    # Check disjointness with diffB
    if not new_diffs_A & diffB:
        # Check Sidon property for A + [x]
        new_sums_A = set(sumA)
        for a in A:
            new_sums_A.add(a + x)
        new_sums_A.add(2 * x)
        if len(new_sums_A) == len(sumA) + len(A) + 1:
            return (0, new_diffs_A, new_sums_A)
    
    # Try adding to B
    new_diffs_B = set()
    for b in B:
        d = x - b
        if d != 0:
            new_diffs_B.add(d)
        d = b - x
        if d != 0:
            new_diffs_B.add(d)
    # Check disjointness with diffA
    if not new_diffs_B & diffA:
        # Check Sidon property for B + [x]
        new_sums_B = set(sumB)
        for b in B:
            new_sums_B.add(b + x)
        new_sums_B.add(2 * x)
        if len(new_sums_B) == len(sumB) + len(B) + 1:
            return (1, new_diffs_B, new_sums_B)
    
    return None


def construct():
    t0 = time.time()
    best_A = []
    best_B = []
    best_score = 0
    
    random.seed(42)
    
    # Strategy: Greedy construction with multiple random restarts
    # Use a deadline guard to ensure we return within time limit
    
    while time.time() - t0 < DEADLINE:
        # Random restart with different initial seeds
        A = []
        B = []
        diffA = set()
        diffB = set()
        sumA = set()
        sumB = set()
        
        # Generate random candidate elements
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - t0 > DEADLINE - 2:
                break
            if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
                break
            
            result = _try_add(A, B, x, diffA, diffB, sumA, sumB)
            if result is not None:
                side, new_diffs, new_sums = result
                if side == 0:
                    A.append(x)
                    diffA = diffA | new_diffs
                    sumA = new_sums
                else:
                    B.append(x)
                    diffB = diffB | new_diffs
                    sumB = new_sums
        
        # Update best if improved
        current_score = len(A) + len(B)
        if current_score > best_score:
            best_score = current_score
            best_A = A[:]
            best_B = B[:]
        
        # If we reached maximum possible, stop early
        if best_score == 2 * MAX_EACH:
            break
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
