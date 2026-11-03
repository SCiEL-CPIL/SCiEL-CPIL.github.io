"""Kissing number in dimension 11: SA on candidate vectors with entries in {-1,0,1}."""
import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 140.0


# EVOLVE-BLOCK-START

def _generate_candidates():
    """Generate all nonzero vectors in {-1,0,1}^11 with at least 2 non-zero entries."""
    candidates = []
    for k in range(2, D + 1):
        for positions in itertools.combinations(range(D), k):
            for signs in itertools.product([1, -1], repeat=k):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidates.append(tuple(v))
    return candidates


def _build_conflict(vectors):
    """Two vectors conflict if their Euclidean distance < max norm in the set.
    For now, use threshold: distance < 2 (since norm-2 vectors are the reference).
    Actually, the task says: min pairwise distance >= max vector norm.
    So we need: for all i!=j, ||v_i - v_j|| >= max_i ||v_i||.
    This is a global constraint. For SA, we can use a local proxy: 
    conflict if ||v_i - v_j||^2 < max_norm^2.
    But max_norm depends on the set. For vectors in {-1,0,1}, max norm is sqrt(11).
    Let's use a fixed threshold: distance >= sqrt(11) ~ 3.317.
    Actually, let's just compute the actual constraint for a given set.
    """
    n = len(vectors)
    V = np.array(vectors, dtype=np.float64)
    norms = np.sqrt(np.sum(V**2, axis=1))
    max_norm = np.max(norms)
    # Pairwise distances
    diff = V[:, np.newaxis, :] - V[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    conflict = dists_sq < max_norm**2 - 1e-9
    np.fill_diagonal(conflict, False)
    return conflict


def _greedy_is(conflict, n, rng):
    """Greedy independent set."""
    selected = []
    available = list(range(n))
    rng.shuffle(available)
    for i in available:
        if all(not conflict[i, j] for j in selected):
            selected.append(i)
    return set(selected)


def _sa_independent_set(conflict, n, time_budget, seed=42):
    rng = random.Random(seed)
    start = time.time()
    
    # Start with greedy
    current = _greedy_is(conflict, n, rng)
    best = set(current)
    best_size = len(current)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start < time_budget:
        move = rng.random()
        if move < 0.5:
            # Add a node
            candidates = [i for i in range(n) if i not in current]
            if not candidates:
                continue
            i = rng.choice(candidates)
            conflicts_with = [j for j in current if conflict[i, j]]
            if conflicts_with:
                continue
            new_current = current | {i}
            delta = 1
        elif move < 0.8:
            # Remove a node
            if not current:
                continue
            i = rng.choice(list(current))
            new_current = current - {i}
            delta = -1
        else:
            # Swap
            if not current:
                continue
            i = rng.choice(list(current))
            candidates = [j for j in range(n) if j not in current and j != i]
            if not candidates:
                continue
            j = rng.choice(candidates)
            remaining = current - {i}
            if all(not conflict[j, k] for k in remaining):
                new_current = remaining | {j}
                delta = 0
            else:
                continue
        
        if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = new_current
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        
        T = max(T * cooling, T_min)
    
    return list(best)


def construct():
    start = time.time()
    candidates = _generate_candidates()
    n = len(candidates)
    
    # Use a fixed threshold for conflict: distance < 2 (norm-2 reference)
    # But the actual constraint is distance >= max_norm of the selected set.
    # For vectors in {-1,0,1}, max possible norm is sqrt(11).
    # Let's use threshold: distance^2 < 4 (i.e., distance < 2) as a conservative proxy.
    # This ensures that if all pairs have distance >= 2, and max_norm <= 2, we're valid.
    # But max_norm can be up to sqrt(11). So this proxy is too strict.
    
    # Better: use the actual constraint. For a candidate set, max_norm = sqrt(k_max) where k_max is max number of nonzeros.
    # Let's just use distance^2 < 11 as conflict (since max possible norm^2 = 11).
    # This is a necessary condition: if dist^2 < 11, it might violate. If dist^2 >= 11, it's safe.
    
    V = np.array(candidates, dtype=np.float64)
    # Precompute pairwise distance squared
    diff = V[:, np.newaxis, :] - V[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Conflict if dist^2 < 11 (max possible norm^2)
    conflict = dists_sq < 11 - 1e-9
    np.fill_diagonal(conflict, False)
    
    time_left = DEADLINE - (time.time() - start) - 5.0
    if time_left < 1.0:
        # Not enough time, return empty or small set
        return np.array([candidates[0]], dtype=np.int64)
    
    indices = _sa_independent_set(conflict, n, time_left, seed=42)
    
    result = np.array([candidates[i] for i in indices], dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
