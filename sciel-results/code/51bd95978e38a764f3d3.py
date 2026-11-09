# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than or equal to their minimum pairwise distance.
    
    The seed solution uses 220 points (all vectors with exactly two ±1 entries, norm²=2).
    We aim to exceed 220 by searching over larger candidate pools with greedy + local improvement.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill

    d = 11

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # --- Generate candidates with coordinates in {-2,-1,0,1,2} and norm_sq <= 6 ---
    # We'll be more targeted: generate candidates by norm_sq.
    
    def gen_vectors_with_coords(d, coord_set):
        """Generate all vectors with coordinates in coord_set."""
        for combo in product(coord_set, repeat=d):
            yield np.array(combo, dtype=np.int64)

    # We want to be smart about which candidates to include.
    # Let's generate candidates with small L2 norm.
    
    # Strategy: generate all integer vectors with |coord| <= 2 and norm_sq <= 8
    # But that's 5^11 = 48.8M vectors. Too many to generate all.
    
    # Better: generate by number of non-zero coordinates.
    # k non-zero coords, each ±1 or ±2.
    
    def generate_candidates_up_to(d, max_nonzero, max_coord_abs):
        """
        Generate integer vectors in R^d with at most max_nonzero non-zero coordinates,
        each coordinate in [-max_coord_abs, max_coord_abs] \ {0} for non-zero.
        """
        cands = []
        for k in range(1, max_nonzero + 1):
            for indices in combinations(range(d), k):
                # All sign/value combinations for these k positions
                vals_range = range(1, max_coord_abs + 1)
                for vals in product(vals_range, repeat=k):
                    for signs in product([1, -1], repeat=k):
                        p = np.zeros(d, dtype=np.int64)
                        for i, idx in enumerate(indices):
                            p[idx] = signs[i] * vals[i]
                        cands.append(p)
        return cands

    # Let's try max_nonzero=3, max_coord_abs=2
    # k=1: C(11,1)*2*2 = 44
    # k=2: C(11,2)*2*2*2*2 = 55*16 = 880
    # k=3: C(11,3)*2*2*2*2*2*2 = 165*64 = 10560
    # Total: ~11484 candidates. Manageable.
    
    candidates = generate_candidates_up_to(d, 3, 2)
    # Remove duplicates (shouldn't be any, but just in case)
    # Sort by norm_sq
    norms = np.array([np.dot(p, p) for p in candidates], dtype=np.int64)
    order = np.argsort(norms, kind='stable')
    candidates = [candidates[i] for i in order]
    norms = norms[order]

    # Also, we want to include the known good set of norm-2 points.
    # They're already included (k=2, vals=(1,1), all signs).
    
    # --- Greedy construction with multiple random orderings ---
    
    def greedy_construct(cand_list, cand_norms, max_time=10.0):
        """
        Greedy: iterate candidates, add if compatible.
        """
        t0 = time.time()
        S = np.array([], dtype=np.int64).reshape(0, d)
        max_norm_sq = 0
        
        for i, p in enumerate(cand_list):
            if time.time() - t0 > max_time:
                break
            pn = cand_norms[i]
            if len(S) == 0:
                S = p.reshape(1, -1).copy()
                max_norm_sq = pn
                continue
            new_max = max(max_norm_sq, pn)
            # Check all distances
            diffs = S - p
            dists = np.sum(diffs * diffs, axis=1)
            if np.min(dists) >= new_max:
                S = np.vstack([S, p])
                max_norm_sq = new_max
        return S, max_norm_sq

    # Try greedy with different candidate orderings
    best_S = np.array([], dtype=np.int64).reshape(0, d)
    best_count = 0
    
    # Ordering 1: by norm (original)
    S1, mn1 = greedy_construct(candidates, norms, max_time=min(15.0, remaining() * 0.15))
    if len(S1) > best_count:
        best_S = S1
        best_count = len(S1)

    # Ordering 2: random shuffles (a few)
    rng = np.random.RandomState(42)
    for trial in range(3):
        if remaining() < 30:
            break
        idx = np.arange(len(candidates))
        rng.shuffle(idx)
        cands_shuffled = [candidates[i] for i in idx]
        norms_shuffled = norms[idx]
        S, mn = greedy_construct(cands_shuffled, norms_shuffled, max_time=min(10.0, remaining() * 0.12))
        if len(S) > best_count:
            best_S = S
            best_count = len(S)

    # --- Local improvement: try to add more candidates not in best_S ---
    if best_count > 0 and remaining() > 20:
        # Try adding candidates that weren't selected
        S = best_S.copy()
        max_norm_sq = int(np.max(np.sum(S * S, axis=1)))
        in_set = np.zeros(len(candidates), dtype=bool)
        # Mark which candidates are in S
        for i, p in enumerate(candidates):
            # Check if p is in S (exact match)
            # This is slow; let's use a different approach
            pass
        
        # Better: for each candidate not obviously in S, try to add
        # We'll re-check by trying to add each candidate
        added_any = True
        while added_any and remaining() > 10:
            added_any = False
            for i, p in enumerate(candidates):
                if remaining() < 10:
                    break
                pn = norms[i]
                new_max = max(max_norm_sq, pn)
                diffs = S - p
                dists = np.sum(diffs * diffs, axis=1)
                if np.min(dists) >= new_max:
                    S = np.vstack([S, p])
                    max_norm_sq = new_max
                    added_any = True
            # One pass at a time
        
        if len(S) > best_count:
            best_S = S
            best_count = len(S)

    # --- Try a different structural approach: use only norm-2 points but check if we can add norm-1 ---
    # The 220 norm-2 points are a baseline. Let's ensure we have at least that.
    
    norm2_only = [p for p in candidates if norms[list(candidates).index(p)] == 2] if False else None
    # Simpler: generate norm-2 points directly
    norm2_cands = []
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                norm2_cands.append(p)
    
    if len(norm2_cands) > best_count:
        best_S = np.array(norm2_cands, dtype=np.int64)
        best_count = len(best_S)

    # --- Final verification ---
    if len(best_S) == 0:
        # Fallback: 2 points
        best_S = np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Verify
    if len(best_S) >= 2:
        norms_sq = np.sum(best_S ** 2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        
        # Check all pairs
        valid = True
        for i in range(len(best_S)):
            if remaining() < 5:
                break
            for j in range(i + 1, len(best_S)):
                diff = best_S[i] - best_S[j]
                dist_sq = int(np.sum(diff * diff))
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to norm-2 set
            best_S = np.array(norm2_cands, dtype=np.int64)

    return np.around(best_S).astype(np.int64)

# EVOLVE-BLOCK-END
