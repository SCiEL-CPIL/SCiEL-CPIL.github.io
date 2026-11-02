"""Kissing number in dimension 11: maximize number of integer vectors
with min pairwise distance >= max norm.

Uses greedy + local search on conflict graph of small-norm integer vectors.
"""

import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 140  # safety margin before 150s limit


def _generate_candidates():
    """Generate all integer vectors in R^11 with squared norm in {2, 3, 4}."""
    candidates = []
    seen = set()
    
    # Norm squared = 2: two coordinates ±1
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                t = tuple(v)
                if t not in seen:
                    seen.add(t)
                    candidates.append(t)
    
    # Norm squared = 3: three coordinates ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    t = tuple(v)
                    if t not in seen:
                        seen.add(t)
                        candidates.append(t)
    
    # Norm squared = 4: one coordinate ±2, or four coordinates ±1
    for i in range(D):
        for si in (2, -2):
            v = [0] * D
            v[i] = si
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                candidates.append(t)
    
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        t = tuple(v)
                        if t not in seen:
                            seen.add(t)
                            candidates.append(t)
    
    return candidates


def _diff_sq(u, v):
    """Compute squared distance between two vectors."""
    s = 0
    for a, b in zip(u, v):
        d = a - b
        s += d * d
    return s


def _build_conflict_sets(candidates, threshold):
    """Build conflict sets: i and j conflict if 0 < ||u_i - u_j||^2 < threshold."""
    n = len(candidates)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _diff_sq(candidates[i], candidates[j])
            if 0 < dsq < threshold:
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts


def _greedy_is(n, conflicts, rng, strategy="random"):
    """Greedy independent set."""
    available = set(range(n))
    selected = []
    
    while available:
        if strategy == "min_degree":
            best_v = None
            best_deg = float('inf')
            # Sample for speed
            if len(available) > 100:
                sample = rng.sample(list(available), 50)
            else:
                sample = list(available)
            for v in sample:
                deg = len(conflicts[v] & available)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
        else:
            best_v = rng.choice(list(available))
        
        selected.append(best_v)
        available.discard(best_v)
        for nb in conflicts[best_v]:
            available.discard(nb)
    
    return selected


def _local_search(selected_set, n, conflicts, threshold, rng, max_rounds=50):
    """Improve independent set via add/swap operations."""
    selected_set = set(selected_set)
    
    for _ in range(max_rounds):
        improved = False
        
        # Try adding any non-conflicting vertex
        for v in range(n):
            if v not in selected_set and not (conflicts[v] & selected_set):
                selected_set.add(v)
                improved = True
                break
        
        if not improved:
            # Try swap: remove one vertex, add one or more
            for v in list(selected_set):
                # Count how many vertices are blocked by v
                blocked_by_v = conflicts[v] & selected_set
                if not blocked_by_v:
                    continue
                # If we remove v, what can we add?
                temp_set = selected_set - {v}
                new_adds = []
                for w in range(n):
                    if w not in temp_set and not (conflicts[w] & temp_set):
                        new_adds.append(w)
                if len(new_adds) > len(blocked_by_v):
                    selected_set.discard(v)
                    for w in new_adds[:len(blocked_by_v) + 1]:
                        if not (conflicts[w] & selected_set):
                            selected_set.add(w)
                    improved = True
                    break
        
        if not improved:
            break
    
    return selected_set


def _verify(candidates, selected_indices, threshold):
    """Verify the set is valid."""
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            dsq = _diff_sq(candidates[selected_indices[i]], candidates[selected_indices[j]])
            if dsq < threshold:
                return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_count = 0
    best_vectors = None
    
    # Try with threshold = 3 (max norm^2 = 3) and threshold = 4 (max norm^2 = 4)
    for max_norm_sq in [3, 4]:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        threshold = max_norm_sq
        candidates = _generate_candidates()
        n = len(candidates)
        
        # Filter: only keep vectors with norm^2 <= max_norm_sq
        filtered = []
        for c in candidates:
            nsq = sum(x * x for x in c)
            if nsq <= max_norm_sq:
                filtered.append(c)
        candidates = filtered
        n = len(candidates)
        
        conflicts = _build_conflict_sets(candidates, threshold)
        
        # Multiple restarts
        for restart in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            strategy = rng.choice(["random", "min_degree", "random", "min_degree"])
            selected = _greedy_is(n, conflicts, rng, strategy)
            selected_set = set(selected)
            
            # Local search
            selected_set = _local_search(selected_set, n, conflicts, threshold, rng)
            
            # Verify
            sel_list = list(selected_set)
            if _verify(candidates, sel_list, threshold):
                if len(selected_set) > best_count:
                    best_count = len(selected_set)
                    best_vectors = [candidates[i] for i in sel_list]
    
    if best_vectors is None:
        # Fallback: 22 unit vectors won't work since they conflict.
        # Use just a few well-separated vectors
        vectors = []
        for i in range(D):
            v = [0] * D
            v[i] = 1
            vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
