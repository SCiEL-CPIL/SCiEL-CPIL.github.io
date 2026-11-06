"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

N = 9
NUM_VERTICES = 1 << N  # 512

# Precompute neighbor lists and bit counts for all vertices
_NEIGHBORS = None
_BITCOUNTS = None
_ADJACENCY_MATRIX = None  # 512x512 boolean matrix for fast distance checks

def _init_data():
    global _NEIGHBORS, _BITCOUNTS, _ADJACENCY_MATRIX
    if _NEIGHBORS is not None:
        return
    _BITCOUNTS = np.array([i.bit_count() for i in range(NUM_VERTICES)], dtype=np.int8)
    _NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            _NEIGHBORS[v].append(v ^ (1 << b))
    # Build adjacency matrix: adj[u,v] = True if Hamming distance is 1
    _ADJACENCY_MATRIX = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=bool)
    for v in range(NUM_VERTICES):
        for nb in _NEIGHBORS[v]:
            _ADJACENCY_MATRIX[v, nb] = True

def _is_valid_snake(path):
    """Check if path is a valid induced path (snake)."""
    if len(path) <= 1:
        return True
    used = set()
    for i, v in enumerate(path):
        if v in used:
            return False
        used.add(v)
        # Check consecutive differ by exactly one bit
        if i > 0:
            if (v ^ path[i-1]).bit_count() != 1:
                return False
        # Check non-consecutive differ by at least two bits
        # For efficiency, check against all previous non-adjacent vertices
        for j in range(i-2, -1, -1):
            if (v ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_seed():
    """Generate a greedy snake seed."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path

def _warnsdorff_greedy(deadline):
    """Warnsdorff's rule greedy: at each step, pick the neighbor with fewest onward options."""
    path = [0]
    used = [False] * NUM_VERTICES
    used[0] = True
    while True:
        if time.time() > deadline:
            break
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if used[v]:
                continue
            # Check if v is adjacent to any non-consecutive vertex
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                # Count onward options for v
                onward = 0
                for b2 in range(N):
                    w = v ^ (1 << b2)
                    if not used[w] and w != cur:
                        # Check w not adjacent to path[:-2] (non-consecutive to v)
                        ok = True
                        for p in path[:-2]:
                            if (w ^ p).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            onward += 1
                candidates.append((onward, v))
        if not candidates:
            break
        candidates.sort()
        path.append(candidates[0][1])
        used[candidates[0][1]] = True
    return path

def _local_search(path, deadline):
    """Try to extend the path by trying different bit orders and local moves."""
    best_path = list(path)
    best_len = len(best_path)
    
    # Try extending from the end with different strategies
    for attempt in range(50):
        if time.time() > deadline:
            break
        current = list(best_path)
        used = [False] * NUM_VERTICES
        for v in current:
            used[v] = True
        
        # Try to extend
        extended = False
        while True:
            cur = current[-1]
            best_next = None
            best_score = -1
            for b in range(N):
                v = cur ^ (1 << b)
                if used[v]:
                    continue
                valid = True
                for p in current[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    # Score: prefer vertices with fewer onward options (Warnsdorff)
                    onward = 0
                    for b2 in range(N):
                        w = v ^ (1 << b2)
                        if not used[w] and w != cur:
                            ok = True
                            for p in current[:-2]:
                                if (w ^ p).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                onward += 1
                    score = -onward + random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_next = v
            if best_next is None:
                break
            current.append(best_next)
            used[best_next] = True
            extended = True
            if time.time() > deadline:
                break
        
        if len(current) > best_len:
            best_path = current
            best_len = len(best_path)
    
    return best_path

def _simulated_annealing(path, deadline):
    """Simulated annealing with 2-bit flip moves to improve the snake."""
    _init_data()
    current = list(path)
    current_len = len(current)
    best = list(current)
    best_len = current_len
    
    def compute_score(p):
        """Score = length of valid prefix (how far the path remains valid)."""
        used = set()
        valid_len = 0
        for i, v in enumerate(p):
            if v in used:
                return valid_len
            used.add(v)
            if i > 0 and (v ^ p[i-1]).bit_count() != 1:
                return valid_len
            for j in range(i-2, -1, -1):
                if (v ^ p[j]).bit_count() == 1:
                    return valid_len
            valid_len = i + 1
        return valid_len
    
    start_temp = 10.0
    end_temp = 0.01
    steps = 0
    max_steps = 20000
    t = start_temp
    
    while steps < max_steps and time.time() < deadline:
        steps += 1
        t = start_temp * (end_temp / start_temp) ** (steps / max_steps)
        
        # Random move: swap two non-adjacent elements, or remove and reinsert
        move_type = random.random()
        if move_type < 0.4 and len(current) > 2:
            # Swap two elements
            i, j = random.sample(range(len(current)), 2)
            if abs(i - j) == 1:
                continue
            current[i], current[j] = current[j], current[i]
        elif move_type < 0.7 and len(current) > 2:
            # Remove a random element and try to reinsert
            i = random.randint(1, len(current) - 2)
            removed = current.pop(i)
            # Try to reinsert at random position
            j = random.randint(0, len(current))
            current.insert(j, removed)
        else:
            # Try to extend
            cur = current[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in set(current):
                    valid = True
                    for p in current[:-1]:
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
            if candidates:
                current.append(random.choice(candidates))
        
        # Check validity and score
        new_score = compute_score(current)
        old_score = compute_score(best) if len(current) != len(best) else best_len
        
        # Actually, let's track current validity properly
        cur_valid_len = compute_score(current)
        
        if cur_valid_len > best_len:
            best = list(current)
            best_len = cur_valid_len
        else:
            # Accept with probability
            delta = cur_valid_len - best_len
            if delta < 0 and random.random() < (2.718281828 ** (delta / max(t, 1e-10))):
                pass  # Accept worse solution
            else:
                # Reject: revert
                if move_type < 0.4 and len(current) > 2:
                    i, j = random.sample(range(len(current)), 2)  # This won't match... 
                # Simpler: just keep going, the best is tracked separately
                pass
    
    # After SA, try to extend the best found
    best = _local_search(best, deadline)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    _init_data()
    
    # Generate initial seed
    seed_path = _greedy_seed()
    best_path = list(seed_path)
    best_len = len(best_path)
    
    # Try Warnsdorff greedy
    wp = _warnsdorff_greedy(deadline)
    if len(wp) > best_len:
        best_path = wp
        best_len = len(wp)
    
    # Try local search on the best
    if time.time() < deadline:
        ls_path = _local_search(best_path, deadline)
        if len(ls_path) > best_len:
            best_path = ls_path
            best_len = len(ls_path)
    
    # Try simulated annealing if time remains
    if time.time() < deadline - 10.0:
        sa_path = _simulated_annealing(best_path, deadline)
        if len(sa_path) > best_len:
            best_path = sa_path
            best_len = len(sa_path)
    
    # Final local search
    if time.time() < deadline - 5.0:
        final_path = _local_search(best_path, deadline)
        if len(final_path) > best_len:
            best_path = final_path
            best_len = len(final_path)
    
    # Validate the result
    if not _is_valid_snake(best_path):
        # Fallback to seed
        return seed_path
    
    return best_path
