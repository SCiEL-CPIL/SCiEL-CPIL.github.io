"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()
    
    def is_deadline():
        return time.perf_counter() - t0 > DEADLINE_S

    # Generate candidates
    candidates = []
    norms_sq = []
    
    # m=1 (norm_sq = 1)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            norms_sq.append(1)
            
    # m=2 (norm_sq = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                norms_sq.append(2)
                
    # m=3 (norm_sq = 3)
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    norms_sq.append(3)
                    
    # m=4 (norm_sq = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            norms_sq.append(4)
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        norms_sq.append(4)
                        
    V = np.array(candidates, dtype=np.int64)
    NSQ = np.array(norms_sq, dtype=np.int64)
    N = len(candidates)
    
    # Precompute compatibility matrix
    # compat[i,j] = True if V[i] and V[j] can coexist
    # Condition: 2 * V[i] . V[j] <= min(NSQ[i], NSQ[j])
    compat = np.zeros((N, N), dtype=bool)
    block_size = 256
    
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        dots = V[start:end] @ V.T
        min_ns = np.minimum(NSQ[start:end][:, None], NSQ[None, :])
        compat[start:end, :] = (2 * dots <= min_ns)
        
        if is_deadline():
            break
    
    # CSP with backtracking and forward checking
    # Most-constrained-variable heuristic: at each step, pick the variable (vector)
    # with the fewest remaining compatible values (i.e., fewest compatible neighbors in domain)
    
    best_k = 0
    best_set = None
    
    rng = np.random.RandomState(42)
    
    # We'll do multiple runs with different random tie-breaking
    n_restarts = 30
    
    for restart in range(n_restarts):
        if is_deadline():
            break
        
        domain = np.ones(N, dtype=bool)  # remaining candidates
        chosen = []
        chosen_mask = np.zeros(N, dtype=bool)
        
        # Initial domain: all vectors
        # At each step, find the most constrained variable in the domain
        # and add it, then forward-check
        
        while True:
            if is_deadline():
                break
            
            if not domain.any():
                break
            
            # Find the most constrained variable: the one with fewest compatible neighbors in domain
            # For each i in domain, count how many j in domain have compat[i,j] = True
            # We want the one with the MINIMUM count (most constrained)
            
            domain_indices = np.where(domain)[0]
            if len(domain_indices) == 0:
                break
            
            # Count compatible neighbors for each remaining vector
            # This is expensive: O(N^2) per step
            # Optimization: only consider vectors in domain
            sub_compat = compat[np.ix_(domain_indices, domain_indices)]
            compat_counts = sub_compat.sum(axis=1)
            
            # Most constrained: minimum count
            # Tie-break randomly for diversity across restarts
            min_count = compat_counts.min()
            min_indices = domain_indices[compat_counts == min_count]
            
            if restart == 0:
                # Deterministic: pick first
                pick = min_indices[0]
            else:
                pick = min_indices[rng.randint(len(min_indices))]
            
            # Add pick to chosen
            chosen.append(pick)
            chosen_mask[pick] = True
            domain[pick] = False
            
            # Forward checking: remove all vectors incompatible with pick
            # A vector j is incompatible with pick if not compat[pick, j]
            # Remove them from domain
            incompatible = domain & ~compat[pick, :]
            domain[incompatible] = False
            
            # If no vectors remain, break
            if not domain.any():
                break
        
        k = len(chosen)
        if k > best_k:
            best_k = k
            best_set = chosen[:k]
        
        if is_deadline():
            break
    
    if best_k > 0:
        return V[np.array(best_set)]
    else:
        # Fallback
        base = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    base.append(v)
        return np.array(base, dtype=np.int64)
