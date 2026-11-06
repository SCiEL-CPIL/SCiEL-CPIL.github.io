"""
Snake-in-the-Box in Q9 using Warnsdorff's rule with backtracking and randomized restarts.

Structural change: Replace MILP with a heuristic backtracking search that uses
Warnsdorff's rule (prefer moves to neighbors with fewest available extensions)
combined with randomized restarts. This exploits the local structure of the
hypercube far more effectively than a global linear programming formulation.
"""
import time
import random
import sys
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()
_NEIGHBOR_SETS = [set(n) for n in _NEIGHBORS]

def _is_valid_extension(v, used_set, current):
    """Check if vertex v can be added after current in the path.
    v must not be in used_set, and no neighbor of v (other than current) can be in used_set."""
    if v in used_set:
        return False
    for w in _NEIGHBORS[v]:
        if w in used_set and w != current:
            return False
    return True

def _warnsdorff_score(v, used_set, current):
    """Score a candidate move: count how many valid extensions v has.
    Lower score = more constrained = preferred (Warnsdorff's rule)."""
    count = 0
    for w in _NEIGHBORS[v]:
        if w not in used_set and w != current:
            # Check if w would be a valid extension from v
            valid = True
            for x in _NEIGHBORS[w]:
                if x in used_set and x != v:
                    valid = False
                    break
            if valid:
                count += 1
    return count

def _backtrack_search(start, used_set, path, max_depth, deadline, rng):
    """Backtracking search with Warnsdorff's rule. Returns the longest path found."""
    if time.time() > deadline:
        return path
    
    if len(path) >= max_depth:
        return path
    
    current = path[-1]
    
    # Get candidate moves: neighbors of current that are valid extensions
    candidates = []
    for v in _NEIGHBORS[current]:
        if _is_valid_extension(v, used_set, current):
            candidates.append(v)
    
    if not candidates:
        return path
    
    # Score candidates using Warnsdorff's rule
    scored = [( _warnsdorff_score(v, used_set, current), rng.random(), v) for v in candidates]
    scored.sort()
    
    best_path = list(path)
    
    for _, _, v in scored:
        if time.time() > deadline:
            break
        
        used_set.add(v)
        path.append(v)
        
        result = _backtrack_search(start, used_set, path, max_depth, deadline, rng)
        if len(result) > len(best_path):
            best_path = result
        
        path.pop()
        used_set.discard(v)
    
    return best_path

def _greedy_warnsdorff(start, deadline, rng):
    """Greedy construction using Warnsdorff's rule (no backtracking)."""
    path = [start]
    used_set = {start}
    
    while time.time() < deadline:
        current = path[-1]
        candidates = []
        for v in _NEIGHBORS[current]:
            if _is_valid_extension(v, used_set, current):
                candidates.append(v)
        
        if not candidates:
            break
        
        # Score and pick best
        scored = [( _warnsdorff_score(v, used_set, current), rng.random(), v) for v in candidates]
        scored.sort()
        _, _, v = scored[0]
        path.append(v)
        used_set.add(v)
    
    return path

def _greedy_extend(path, used_set=None, deadline=None):
    """Extend a path greedily from both ends using Warnsdorff's rule."""
    if used_set is None:
        used_set = set(path)
    if deadline is None:
        deadline = time.time() + 5.0
    
    # Try extending from the end
    while time.time() < deadline:
        cur = path[-1]
        candidates = []
        for v in _NEIGHBORS[cur]:
            if _is_valid_extension(v, used_set, cur):
                candidates.append(v)
        if not candidates:
            break
        scored = [(_warnsdorff_score(v, used_set, cur), random.random(), v) for v in candidates]
        scored.sort()
        v = scored[0][2]
        path.append(v)
        used_set.add(v)
    
    # Try extending from the beginning
    while time.time() < deadline:
        cur = path[0]
        candidates = []
        for v in _NEIGHBORS[cur]:
            if _is_valid_extension(v, used_set, cur):
                candidates.append(v)
        if not candidates:
            break
        scored = [(_warnsdorff_score(v, used_set, cur), random.random(), v) for v in candidates]
        scored.sort()
        v = scored[0][2]
        path.insert(0, v)
        used_set.add(v)
    
    return path

def _try_insert(path, vertex, used_set):
    """Try to insert a vertex into the path at a valid position."""
    if vertex in used_set:
        return None
    
    # Try inserting between consecutive vertices
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        if (vertex ^ u).bit_count() == 1 and (vertex ^ v).bit_count() == 1:
            # Check induced path property
            valid = True
            for w in _NEIGHBORS[vertex]:
                if w in used_set and w != u and w != v:
                    valid = False
                    break
            if valid:
                new_path = path[:i+1] + [vertex] + path[i+1:]
                return new_path
    
    # Try prepending
    if (vertex ^ path[0]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w in used_set and w != path[0]:
                valid = False
                break
        if valid:
            return [vertex] + path
    
    # Try appending
    if (vertex ^ path[-1]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w in used_set and w != path[-1]:
                valid = False
                break
        if valid:
            return path + [vertex]
    
    return None

def _local_improve(path, deadline, rng):
    """Improve path via random vertex insertion attempts."""
    used_set = set(path)
    best_path = list(path)
    best_len = len(best_path)
    
    # Build list of unused vertices
    unused = [v for v in range(TOTAL_VERTS) if v not in used_set]
    rng.shuffle(unused)
    
    for v in unused:
        if time.time() > deadline - 2.0:
            break
        new_path = _try_insert(best_path, v, used_set)
        if new_path:
            best_path = new_path
            used_set = set(new_path)
            best_len = len(new_path)
    
    return best_path

def construct():
    deadline = time.time() + 145.0
    rng = random.Random(12345)
    
    best_path = None
    best_len = 0
    
    # Phase 1: Multiple greedy Warnsdorff constructions from random starts
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline - 30.0:
            break
        start = rng.randint(0, TOTAL_VERTS - 1)
        path = _greedy_warnsdorff(start, deadline - 30.0, rng)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Phase 2: Backtracking search from best greedy start
    if best_path and time.time() < deadline - 20.0:
        start = best_path[0]
        used_set = set(best_path)
        path = list(best_path)
        # Try backtracking to extend
        result = _backtrack_search(start, used_set, path, len(path) + 50, deadline - 10.0, rng)
        if len(result) > best_len:
            best_path = result
            best_len = len(result)
    
    # Phase 3: More greedy restarts with different random seeds
    for i in range(15):
        if time.time() > deadline - 15.0:
            break
        start = rng.randint(0, TOTAL_VERTS - 1)
        path = _greedy_warnsdorff(start, deadline - 15.0, rng)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    # Phase 4: Greedy extension and local improvement
    if best_path:
        if time.time() < deadline - 5.0:
            best_path = _greedy_extend(best_path, deadline=deadline - 5.0)
            best_len = len(best_path)
        
        if time.time() < deadline - 2.0:
            best_path = _local_improve(best_path, deadline - 2.0, rng)
            best_len = len(best_path)
    
    # Final safety: ensure we have a valid path
    if best_path is None:
        # Fallback: simple greedy from 0
        best_path = [0]
        used_set = {0}
        while True:
            cur = best_path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if _is_valid_extension(v, used_set, cur):
                    best_path.append(v)
                    used_set.add(v)
                    found = True
                    break
            if not found:
                break
    
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
