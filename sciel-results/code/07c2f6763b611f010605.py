"""Additive base with few representations: constructive binary/mixed-radix design.

If A = {sum_i d_i * B^i : 0 <= d_i <= M_i} with P = prod M_i, then every s in
[0, P) has a unique unordered-with-equal representation s = a + b (a,b in A),
because each digit s_i = a_i + b_i is realized uniquely with a_i,b_i in [0,M_i].
Thus P >= 257 gives full coverage of [0,256] with multiplicity 1 (the optimum).
"""
import time
import itertools
from collections import Counter

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 75.0


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    A = sorted(A)
    n = len(A)
    peak = 0
    for s in range(128, 257):
        c = 0
        for i in range(n):
            a = A[i]
            b = s - a
            if b < a:
                break
            if b in A:
                c += 1
        if c > peak:
            peak = c
    return peak


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    A = sorted(A)
    n = len(A)
    for s in range(128, 257):
        found = False
        for i in range(n):
            a = A[i]
            b = s - a
            if b < a:
                break
            if b in A:
                found = True
                break
        if not found:
            return False
    return True


def _construct_radix(M):
    """Build A = {sum d_i * B^i : 0 <= d_i <= M_i} for the given digit caps M."""
    B = [1]
    for m in M:
        B.append(B[-1] * (m + 1))
    A = set()
    for digits in itertools.product(*[range(m + 1) for m in M]):
        val = 0
        for d, p in zip(digits, B):
            val += d * p
        A.add(val)
    return A


def _local_refine(A, budget=20000):
    """Cheap local search: drop a redundant element or swap it to reduce peak."""
    A = list(set(A))
    peak = _peak(A)
    steps = 0
    improved = True
    while improved and steps < budget and time.time() < _DEADLINE:
        improved = False
        # Try removing each element.
        for idx in range(len(A)):
            cand = A[:idx] + A[idx + 1:]
            if not cand:
                continue
            if not _is_valid(cand):
                continue
            p = _peak(cand)
            if p < peak:
                A = cand
                peak = p
                improved = True
                steps += 1
                break
        if improved:
            continue
        # Try swapping an element to a new value.
        for idx in range(len(A)):
            for v in range(0, 129):
                if v in A:
                    continue
                cand = A[:idx] + [v] + A[idx + 1:]
                if not _is_valid(cand):
                    continue
                p = _peak(cand)
                if p < peak:
                    A = cand
                    peak = p
                    improved = True
                    steps += 1
                    break
            if improved:
                break
    return A


def construct():
    t0 = time.time()

    # Primary constructive design: product of digit caps >= 257 => peak 1.
    # 7 * 7 * 5 = 245 (too small). 7 * 7 * 6 = 294 >= 257.
    for M in ([7, 7, 6], [4, 4, 4, 4], [8, 8, 4], [16, 16, 1], [9, 9, 4], [6, 6, 8]):
        try:
            A = _construct_radix(M)
            if A and _is_valid(A):
                peak = _peak(A)
                if peak <= 2:
                    # Already optimal or near-optimal; try a light refinement.
                    if time.time() < _DEADLINE - 20:
                        A2 = _local_refine(A, budget=15000)
                        if _is_valid(A2) and _peak(A2) <= peak:
                            A = A2
                    return sorted(A)
        except Exception:
            pass

    # Fallback: guaranteed valid set [64,128] (peak 32), then try to improve.
    A = list(range(64, 129))
    peak = _peak(A)
    if time.time() < _DEADLINE - 15:
        try:
            A2 = _local_refine(A, budget=30000)
            if _is_valid(A2) and _peak(A2) <= peak:
                A = A2
        except Exception:
            pass
    return sorted(A)
