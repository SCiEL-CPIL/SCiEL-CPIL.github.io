"""Snake-in-the-box Q_9: maximize induced path length."""
import time
import random
import numpy as np

N = 9
NUM_V = 1 << N  # 512

# Precompute bit counts and single-bit neighbors
_BIT_COUNT = np.array([i.bit_count() for i in range(NUM_V)], dtype=np.int8)
# neighbors[v] = list of v^(1<<b)
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_V)]

# For fast induced-path check: a vertex u can be appended after current path P
# iff dist(u, P[-1]) == 1 and dist(u, x) >= 2 for all x in P[:-1].
# Equivalently: u not in P, and no vertex in P (except last) is at dist 1 from u.
# Since dist(u, x) >= 2 for all x in P[:-1] means u is NOT a neighbor of any x in P[:-1].

def _valid_append(u, path_set, last):
    """Check if u can be appended: dist(u,last)==1 and u not adjacent to any earlier vertex."""
    if u in path_set:
        return False
    # Check u is not a neighbor of any vertex in path[:-1]
    # We need a way to check this fast. Precompute adjacency sets? 512*9 = 4608 edges, fine.
    for nb in _NEIGHBORS[u]:
        if nb != last and nb in path_set:
            return False
    return True


def _greedy_extend(path, path_set, start_bit_order=None, rng=None):
    """Extend path greedily. Returns extended path."""
    if start_bit_order is None:
        start_bit_order = list(range(N))
    if rng is None:
        rng = random.Random(42)
    
    cur = path[-1]
    while True:
        cur = path[-1]
        candidates = []
        for b in start_bit_order:
            v = cur ^ (1 << b)
            if v in path_set:
                continue
            # Check v is not adjacent to any earlier vertex
            ok = True
            for nb in _NEIGHBORS[v]:
                if nb != cur and nb in path_set:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        if not candidates:
            break
        # Pick first candidate (greedy) or random
        if rng is not None:
            v = rng.choice(candidates)
        else:
            v = candidates[0]
        path.append(v)
        path_set.add(v)
    return path


def _random_restarts(num_restarts, time_limit):
    """Try multiple random greedy extensions with different bit orders."""
    best_path = None
    best_len = 0
    start_time = time.time()
    
    for trial in range(num_restarts):
        if time.time() - start_time > time_limit * 0.8:
            break
        rng = random.Random(trial * 12345 + 67890)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path = [0]
        path_set = {0}
        path = _greedy_extend(path, path_set, bit_order, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    return best_path if best_path else None


def _local_search(path, time_limit):
    """Try to improve path via 2-opt style moves: remove a vertex and re-insert elsewhere."""
    # This is complex for induced paths. Instead, try targeted extensions.
    # Try: for each position i, remove path[i], then try to extend from new end.
    # Actually simpler: try adding a vertex that was skipped.
    # Let's try a different approach: for each gap in the path, try to insert a vertex.
    
    best_path = list(path)
    best_len = len(path)
    start_time = time.time()
    
    # Try inserting vertices between consecutive pairs
    path_set = set(path)
    changed = True
    while changed and time.time() - start_time < time_limit * 0.7:
        changed = False
        new_path = list(path)
        new_set = set(path)
        # Try inserting at each gap
        for i in range(len(new_path) - 1):
            a, b = new_path[i], new_path[i+1]
            # Find vertices v such that dist(v,a)==1, dist(v,b)==1, and v not adjacent to any other path vertex
            # v must be a common neighbor of a and b
            common = set(_NEIGHBORS[a]) & set(_NEIGHBORS[b])
            for v in common:
                if v in new_set:
                    continue
                # Check v not adjacent to any other vertex in path
                ok = True
                for nb in _NEIGHBORS[v]:
                    if nb not in (a, b) and nb in new_set:
                        ok = False
                        break
                if ok:
                    new_path.insert(i+1, v)
                    new_set.add(v)
                    changed = True
                    break
            if changed:
                break
        if changed:
            path = new_path
            path_set = new_set
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    return best_path


def _beam_search(width, time_limit):
    """Beam search over partial paths."""
    start_time = time.time()
    # State: (path_tuple, path_set_frozenset) - too heavy. Use path list + set.
    # Beam: list of (path, path_set)
    beam = [([0], {0})]
    best_path = [0]
    best_len = 1
    
    while time.time() - start_time < time_limit * 0.85:
        new_beam = []
        for path, pset in beam:
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in pset:
                    continue
                ok = True
                for nb in _NEIGHBORS[v]:
                    if nb != cur and nb in pset:
                        ok = False
                        break
                if ok:
                    candidates.append(v)
            for v in candidates:
                new_path = path + [v]
                new_set = pset | {v}
                new_beam.append((new_path, new_set))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)
        
        if not new_beam:
            break
        
        # Keep top `width` by length (all same length here since we extend by 1)
        # Actually all new paths have same length = len(beam[0][0]) + 1
        # Sort by some heuristic: prefer paths with more future options?
        # For simplicity, keep random subset or all if width allows
        if len(new_beam) > width:
            # Keep diverse paths: sample
            random.shuffle(new_beam)
            new_beam = new_beam[:width]
        
        beam = new_beam
    
    return best_path


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Phase 1: Greedy with different bit orders (fast)
    best_path = None
    best_len = 0
    
    # Standard greedy
    path = [0]
    pset = {0}
    path = _greedy_extend(path, pset, list(range(N)), None)
    best_len = len(path)
    best_path = list(path)
    
    # Random restarts
    remaining = deadline - time.time()
    if remaining > 2:
        rp = _random_restarts(200, remaining * 0.3)
        if rp and len(rp) > best_len:
            best_len = len(rp)
            best_path = rp
    
    # Phase 2: Local search on best path
    remaining = deadline - time.time()
    if remaining > 5:
        lp = _local_search(best_path, remaining * 0.4)
        if len(lp) > best_len:
            best_len = len(lp)
            best_path = lp
    
    # Phase 3: Beam search if time remains
    remaining = deadline - time.time()
    if remaining > 10:
        bp = _beam_search(50, remaining * 0.7)
        if len(bp) > best_len:
            best_len = len(bp)
            best_path = bp
    
    # Final local search on best found
    remaining = deadline - time.time()
    if remaining > 3:
        lp = _local_search(best_path, remaining * 0.8)
        if len(lp) > best_len:
            best_len = len(lp)
            best_path = lp
    
    # Validate
    if not best_path:
        return [0]
    
    # Final validation
    pset = set(best_path)
    if len(pset) != len(best_path):
        # Duplicates - shouldn't happen but safeguard
        best_path = list(pset)
    
    for i in range(len(best_path) - 1):
        if (best_path[i] ^ best_path[i+1]).bit_count() != 1:
            # Invalid - return greedy baseline
            path = [0]
            pset = {0}
            path = _greedy_extend(path, pset, list(range(N)), None)
            return path
    
    for i in range(len(best_path)):
        for j in range(i+2, len(best_path)):
            if (best_path[i] ^ best_path[j]).bit_count() == 1:
                # Invalid - return greedy baseline
                path = [0]
                pset = {0}
                path = _greedy_extend(path, pset, list(range(N)), None)
                return path
    
    return best_path
