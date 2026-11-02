"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def _generate_candidates(max_weight=5):
    """Generate candidate integer vectors of various weights."""
    candidates = []
    # Weight 2: choose 2 positions, assign +/-1
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[positions[0]] = signs[0]
            v[positions[1]] = signs[1]
            candidates.append(v)
    # Weight 3: choose 3 positions, assign +/-1
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(positions, signs):
                v[p] = s
            candidates.append(v)
    # Weight 4: choose 4 positions, assign +/-1
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(positions, signs):
                v[p] = s
            candidates.append(v)
    # Weight 5: choose 5 positions, assign +/-1
    for positions in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = np.zeros(D, dtype=np.int64)
            for p, s in zip(positions, signs):
                v[p] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)


def _norms_sq(cands):
    """Compute squared norms for each candidate."""
    return np.sum(cands * cands, axis=1)


def _is_compatible(v_new, selected, max_norm_sq):
    """Check if v_new is compatible with all selected vectors given the current max norm squared."""
    if len(selected) == 0:
        return True
    # Constraint: ||v_new - v_sel||^2 >= max_norm_sq for all selected
    # Also need to update max_norm_sq if v_new has larger norm
    new_max = max(max_norm_sq, int(np.dot(v_new, v_new)))
    diffs = selected - v_new
    dists_sq = np.sum(diffs * diffs, axis=1)
    return np.all(dists_sq >= new_max)


def _greedy_select(candidates, norms_sq, rng, time_limit):
    """Greedy stochastic selection of compatible vectors."""
    n = len(candidates)
    selected = []
    max_norm_sq = 0
    
    # Precompute all-pairs distance squared matrix for speed? Too large.
    # Instead, do incremental checks.
    
    # Shuffle candidates for randomization
    order = list(range(n))
    rng.shuffle(order)
    
    start_time = time.time()
    # Try multiple passes to fill in
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        progress = False
        for idx in order:
            v = candidates[idx]
            ns = norms_sq[idx]
            new_max = max(max_norm_sq, ns)
            if len(selected) == 0:
                selected.append(v)
                max_norm_sq = new_max
                progress = True
                continue
            # Check compatibility
            diffs = np.array(selected) - v
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.all(dists_sq >= new_max):
                selected.append(v)
                max_norm_sq = new_max
                progress = True
        if not progress:
            break
    
    return np.array(selected, dtype=np.int64)


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Generate candidate pool
    candidates = _generate_candidates(max_weight=5)
    norms_sq = _norms_sq(candidates)
    n_candidates = len(candidates)
    
    best_count = 0
    best_result = None
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        # Create a random permutation of candidates
        perm = np.random.permutation(n_candidates)
        shuffled_cands = candidates[perm]
        shuffled_norms = norms_sq[perm]
        
        # Greedy selection with this ordering
        selected = []
        max_norm_sq = 0
        start = time.time()
        
        for i in range(n_candidates):
            if time.time() - start > 110:
                break
            v = shuffled_cands[i]
            ns = shuffled_norms[i]
            new_max = max(max_norm_sq, ns)
            if len(selected) == 0:
                selected.append(v)
                max_norm_sq = new_max
                continue
            # Check against all selected
            diffs = np.array(selected) - v
            dists_sq = np.sum(diffs * diffs, axis=1)
            if np.all(dists_sq >= new_max):
                selected.append(v)
                max_norm_sq = new_max
        
        count = len(selected)
        if count > best_count:
            best_count = count
            best_result = np.array(selected, dtype=np.int64)
        
        # If we've already beaten 220 significantly, we can stop early
        if best_count > 300 and time.time() > deadline - 30:
            break
    
    if best_result is None:
        # Fallback: return empty or minimal
        return np.zeros((0, D), dtype=np.int64)
    
    return best_result
# EVOLVE-BLOCK-END
