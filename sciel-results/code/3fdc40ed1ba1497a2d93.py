import numpy as np
import time
import random

N = 128
LO, HI = 128, 256

def compute_stats(A):
    # A: set of ints in [0,128]
    arr = np.zeros(N+1, dtype=np.int64)
    for a in A:
        if 0 <= a <= N:
            arr[a] = 1
    conv = np.convolve(arr, arr, mode='full')
    peak = 0
    covered = True
    Aset = set(A)
    for s in range(LO, HI+1):
        ordered = int(conv[s]) if s < len(conv) else 0
        diag = 1 if (s % 2 == 0 and (s//2) in Aset) else 0
        unord = (ordered + diag) // 2
        if unord == 0:
            covered = False
        if unord > peak:
            peak = unord
    return peak, covered

def construct():
    start = time.time()
    deadline = start + 55  # leave 5s margin
    
    # Initial: seed
    A = set(range(64, N+1))
    peak, covered = compute_stats(A)
    best_peak = peak
    best_A = set(A)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removals
        for a in sorted(A):
            if time.time() > deadline:
                break
            A2 = A - {a}
            if not A2:
                continue
            p, c = compute_stats(A2)
            if c and p < best_peak:
                A = A2
                best_peak = p
                best_A = set(A)
                improved = True
        # Try additions
        for a in range(0, 64):
            if time.time() > deadline:
                break
            if a in A:
                continue
            A2 = A | {a}
            p, c = compute_stats(A2)
            if c and p < best_peak:
                A = A2
                best_peak = p
                best_A = set(A)
                improved = True
        # Try swaps: remove one from A, add one from [0,63]
        if time.time() < deadline:
            for a in sorted(A):
                if time.time() > deadline:
                    break
                for b in range(0, 64):
                    if b in A:
                        continue
                    A2 = (A - {a}) | {b}
                    if not A2:
                        continue
                    p, c = compute_stats(A2)
                    if c and p < best_peak:
                        A = A2
                        best_peak = p
                        best_A = set(A)
                        improved = True
    
    if not best_A:
        return list(range(64, N+1))
    return list(best_A)
