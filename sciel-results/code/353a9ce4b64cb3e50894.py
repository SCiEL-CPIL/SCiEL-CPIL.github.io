"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11
import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START

def _build_candidates():
    """Build candidate pool of nonzero integer vectors in R^11.
    
    We include:
    - Weight-1 vectors: 2*e_i (norm 2) -> 22 vectors
    - Weight-2 vectors: e_i +/- e_j (norm sqrt(2)) -> 440 vectors
    - Weight-3 vectors: e_i +/- e_j +/- e_k (norm sqrt(3)) -> 1320 vectors
    - Weight-4 vectors: e_i +/- e_j +/- e_k +/- e_l (norm 2) -> 3300 vectors
    """
    candidates = []
    
    # Weight-1: 2*e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    # Weight-2: e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Weight-3: e_i +/- e_j +/- e_k
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    # Weight-4: e_i +/- e_j +/- e_k +/- e_l
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates.append(v)
    
    return np.array(candidates, dtype=np.int64)


def construct():
    deadline = time.time() + 110  # safety margin
    
    candidates = _build_candidates()
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # For the kissing number problem, we need:
    # min pairwise distance >= max norm
    # i.e., for all selected vectors a, b: ||a - b|| >= max(||a||, ||b||, ...)
    # Equivalently: ||a-b||^2 >= max_norm_sq
    # where max_norm_sq = max over all selected vectors of ||v||^2
    
    # Strategy: group by norm, then find max independent set in conflict graph
    # Two vectors conflict if their distance is less than the max norm in the set.
    
    # A simpler approach: since we want to maximize count, and the constraint is
    # min_dist >= max_norm, let's try different max_norm values.
    
    # For a given threshold T (squared distance), two vectors are compatible if
    # ||a - b||^2 >= T. We want T = max norm squared in the selected set.
    
    # Let's try: pick a target norm squared T, include only vectors with norm_sq <= T,
    # and require pairwise distance squared >= T.
    
    # Actually, re-reading: "minimum pairwise distance is at least the maximum vector norm"
    # So: min_{a,b in S} ||a-b|| >= max_{v in S} ||v||
    # Squared: min_{a,b} ||a-b||^2 >= max_v ||v||^2
    
    # Let M = max_{v in S} ||v||^2. Then we need all pairwise distances squared >= M.
    # And M is the max norm squared in S.
    
    # Strategy: iterate over possible values of M (which are norm squared values present),
    # filter candidates to those with norm_sq <= M, build conflict graph where two nodes
    # are connected if ||a-b||^2 < M, and find max independent set.
    
    # The norm squared values in our pool: 4, 2, 3, 4 (weight-1: 4, weight-2: 2, weight-3: 3, weight-4: 4)
    # So possible M values: 2, 3, 4
    
    best_set = []
    
    for M in sorted(set(norms_sq.tolist())):
        # Filter: only vectors with norm_sq <= M
        mask = norms_sq <= M
        idx = np.where(mask)[0]
        sub = candidates[idx]
        m = len(sub)
        if m == 0:
            continue
        
        # Compute pairwise distances squared
        # dist^2[i,j] = ||sub[i] - sub[j]||^2
        # = ||sub[i]||^2 + ||sub[j]||^2 - 2*sub[i].sub[j]
        dot = sub @ sub.T
        ns = np.sum(sub * sub, axis=1)
        dist_sq = ns[:, None] + ns[None, :] - 2 * dot
        
        # Conflict: dist_sq[i,j] < M (and i != j)
        # Compatible: dist_sq[i,j] >= M
        conflict = (dist_sq < M).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        
        # Greedy max independent set with degree ordering
        # Degree = number of conflicts
        degrees = conflict.sum(axis=1)
        
        # Sort by degree (ascending), pick greedily
        order = np.argsort(degrees, kind='stable')
        
        selected = []
        remaining = set(range(m))
        
        for idx_pos in order:
            if idx_pos not in remaining:
                continue
            # Check if idx_pos is compatible with all already selected
            compatible = True
            for s in selected:
                if conflict[idx_pos, s] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(idx_pos)
                # Remove all neighbors of idx_pos from remaining
                neighbors = np.where(conflict[idx_pos] == 1)[0]
                for nb in neighbors:
                    remaining.discard(nb)
                remaining.discard(idx_pos)
        
        if len(selected) > len(best_set):
            best_set = selected
        
        if time.time() > deadline - 5:
            break
    
    if not best_set:
        # Fallback: return just weight-1 vectors
        return candidates[:22]
    
    # Map back to original indices
    # We need to track which original candidates were used
    # Re-do: for the best M, get the actual vectors
    
    # Find which M gave best_set
    # Actually let's just recompute for the best case
    best_M = None
    best_sel = []
    for M in sorted(set(norms_sq.tolist())):
        mask = norms_sq <= M
        idx = np.where(mask)[0]
        sub = candidates[idx]
        m = len(sub)
        if m == 0:
            continue
        
        dot = sub @ sub.T
        ns = np.sum(sub * sub, axis=1)
        dist_sq = ns[:, None] + ns[None, :] - 2 * dot
        conflict = (dist_sq < M).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        degrees = conflict.sum(axis=1)
        order = np.argsort(degrees, kind='stable')
        
        selected = []
        remaining = set(range(m))
        for idx_pos in order:
            if idx_pos not in remaining:
                continue
            compatible = True
            for s in selected:
                if conflict[idx_pos, s] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(idx_pos)
                neighbors = np.where(conflict[idx_pos] == 1)[0]
                for nb in neighbors:
                    remaining.discard(nb)
                remaining.discard(idx_pos)
        
        if len(selected) > len(best_sel):
            best_sel = selected
            best_M = M
    
    if not best_sel:
        return candidates[:22]
    
    mask = norms_sq <= best_M
    idx = np.where(mask)[0]
    result = candidates[idx[best_sel]]
    
    return result

# EVOLVE-BLOCK-END
