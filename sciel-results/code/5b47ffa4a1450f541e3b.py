"""
Rewritten program for the second autocorrelation inequality task.
Strategy: Use a Sidon set (B_2 set) to place spikes such that all pairwise differences are distinct,
minimizing autoconvolution sidelobe peaks while maintaining L2 energy.
"""
import time
import numpy as np

N = 50

def _find_sidon_set(n):
    """Find a maximal Sidon set in {0, 1, ..., n-1} using greedy + local search."""
    # Greedy construction
    S = [0]
    for i in range(1, n):
        # Check if i can be added
        ok = True
        for s in S:
            d = abs(i - s)
            for s2 in S:
                if s2 != s and abs(i - s2) == d:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(i)
    
    # Local search to try to improve
    best = S[:]
    for _ in range(100):
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, n):
                if j in best:
                    continue
                # Try swapping best[i] with j
                new_set = best[:i] + [j] + best[i+1:]
                if _is_sidon(new_set):
                    best = new_set
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def _is_sidon(S):
    """Check if S is a Sidon set (all pairwise differences distinct in abs value)."""
    diffs = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = abs(S[i] - S[j])
            if d in diffs:
                return False
            diffs.add(d)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Find a Sidon set
    S = _find_sidon_set(N)
    
    # Create height vector with 1s at Sidon set positions
    h = np.zeros(N)
    for pos in S:
        h[pos] = 1.0
    
    return list(h)
