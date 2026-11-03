import numpy as np
import time
import sys

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 140.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ratio(h):
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def construct():
    # Generate candidate Sidon sets
    # A Sidon set in {0, ..., N-1} has unique sums i+j.
    # We want a set S such that the number of collisions in sums is minimized.
    
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)
    
    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    # Strategy: 
    # 1. Find a good Sidon-like set S.
    # 2. Assign weights w_i to i in S.
    # 3. Optimize w to maximize ratio.
    
    # Heuristic Sidon set: powers of 2 or similar.
    # 1, 2, 4, 8, 16, 32, 64...
    # Indices: 0, 1, 2, 4, 8, 16, 32
    # Sums are unique.
    
    candidate_sets = []
    
    # Set 1: Powers of 2
    s1 = [2**i for i in range(6) if 2**i < N]
    candidate_sets.append(s1)
    
    # Set 2: Random greedy Sidon
    def find_greedy_sidon(n, k):
        S = []
        for _ in range(k):
            best_i = -1
            best_score = -1
            for i in range(n):
                if i in S:
                    continue
                # Check collisions
                collisions = 0
                for j in S:
                    s = i + j
                    # Count how many pairs in S sum to s
                    # Actually, we just want to avoid existing sums
                    pass
                # Simpler: just pick random and check
            # This is slow. Use a known Sidon set or random.
            pass
        return S

    # Use a known construction or random
    np.random.seed(42)
    for _ in range(20):
        # Random sparse set
        k = np.random.randint(5, 12)
        S = np.random.choice(N, k, replace=False)
        candidate_sets.append(S)
        
    # Also try the specific "Sidon" from hint: 1/min_weight
    # This implies making one weight very large, others very small.
    # If S is Sidon, M = h_max^2.
    # R = sum(h_i^2) / (L1 * h_max^2).
    # Let h_max = 1. Others = epsilon.
    # R = (1 + k*eps^2) / ((1 + k*eps) * 1) ~ 1.
    
    # If S is NOT Sidon, but we skew weights, maybe M is dominated by a cross term?
    # No, h_max^2 is always there.
    
    # Let's just optimize the weights on these sets using a simple gradient ascent or random search.
    
    for S in candidate_sets:
        if _time_left() < 10:
            break
        # Initialize weights
        # Try geometric progression
        for r in [0.1, 0.5, 0.9, 10.0, 0.01]:
            h = np.zeros(N)
            for i, idx in enumerate(S):
                h[idx] = r**i
            # Normalize
            h = h / np.sum(h)
            upd(h)
            
            # Try inverse
            h2 = np.zeros(N)
            for i, idx in enumerate(S):
                h2[idx] = r**(-i)
            h2 = h2 / np.sum(h2)
            upd(h2)

    # Final polish on best found
    if _time_left() > 5:
        # Simple local search
        h = best_h.copy()
        for _ in range(50):
            if _time_left() < 5:
                break
            # Perturb
            h_pert = h.copy()
            idx = np.random.randint(N)
            h_pert[idx] += np.random.uniform(-0.1, 0.1)
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) > 0:
                h_pert = h_pert / np.sum(h_pert)
                if _ratio(h_pert) > _ratio(h):
                    h = h_pert
        upd(h)

    return list(best_h)
