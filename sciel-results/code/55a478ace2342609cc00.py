import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 117.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors as a numpy array for speed
    neighbors = np.array([[v ^ (1 << b) for b in range(N)] for v in range(MAX_VERTS)], dtype=np.int32)
    
    best_path = []
    best_len = 0
    
    def can_insert(path, used_mask, candidate, pos):
        """Check if candidate can be inserted at position pos in path.
        pos=0 means prepend, pos=len(path) means append.
        candidate must be adjacent to path[pos-1] (if pos>0) and path[pos] (if pos<len(path)).
        """
        if used_mask[candidate]:
            return False
        # Check adjacency to neighbors in the path
        # Simpler: check that candidate is adjacent to both neighbors
        if pos > 0:
            diff = candidate ^ path[pos-1]
            if diff & (diff - 1) != 0:  # not a power of 2
                return False
        if pos < len(path):
            diff = candidate ^ path[pos]
            if diff & (diff - 1) != 0:
                return False
        # Check that candidate is not adjacent to any other vertex in path
        for u in neighbors[candidate]:
            if u == (path[pos-1] if pos > 0 else -1):
                continue
            if u == (path[pos] if pos < len(path) else -1):
                continue
            if used_mask[u]:
                return False
        return True

    def can_insert_fast(used_mask, candidate, left_neighbor, right_neighbor):
        """Fast check: candidate adjacent to left_neighbor and right_neighbor,
        and not adjacent to any other used vertex."""
        if used_mask[candidate]:
            return False
        if left_neighbor is not None:
            diff = candidate ^ left_neighbor
            if diff & (diff - 1) != 0:
                return False
        if right_neighbor is not None:
            diff = candidate ^ right_neighbor
            if diff & (diff - 1) != 0:
                return False
        for u in neighbors[candidate]:
            if u == left_neighbor or u == right_neighbor:
                continue
            if used_mask[u]:
                return False
        return True

    def greedy_construct(seed_start=None):
        """Build a path using Warnsdorff's heuristic with bidirectional growth."""
        if seed_start is None:
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(neighbors[v0].tolist())
            path = [v0, v1]
        else:
            path = list(seed_start)
        
        used_mask = np.zeros(MAX_VERTS, dtype=bool)
        for v in path:
            used_mask[v] = True
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to extend left
            left_end = path[0]
            valid_left = []
            for v in neighbors[left_end]:
                v = int(v)
                if not used_mask[v]:
                    # Check not adjacent to path[1]
                    if len(path) >= 2:
                        diff = v ^ path[1]
                        if diff & (diff - 1) == 0 and diff != 0:
                            continue
                    # Check not adjacent to any other used
                    ok = True
                    for u in neighbors[v]:
                        u = int(u)
                        if u == left_end:
                            continue
                        if used_mask[u]:
                            ok = False
                            break
                    if ok:
                        valid_left.append(v)
            
            # Try to extend right
            right_end = path[-1]
            valid_right = []
            for v in neighbors[right_end]:
                v = int(v)
                if not used_mask[v]:
                    if len(path) >= 2:
                        diff = v ^ path[-2]
                        if diff & (diff - 1) == 0 and diff != 0:
                            continue
                    ok = True
                    for u in neighbors[v]:
                        u = int(u)
                        if u == right_end:
                            continue
                        if used_mask[u]:
                            ok = False
                            break
                    if ok:
                        valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Warnsdorff: choose end with fewer options
            if len(valid_left) == 0:
                next_v = random.choice(valid_right)
                path.append(next_v)
                used_mask[next_v] = True
            elif len(valid_right) == 0:
                next_v = random.choice(valid_left)
                path.insert(0, next_v)
                used_mask[next_v] = True
            else:
                # Pick from the end with fewer options; if tied, random
                if len(valid_left) <= len(valid_right):
                    next_v = random.choice(valid_left)
                    path.insert(0, next_v)
                    used_mask[next_v] = True
                else:
                    next_v = random.choice(valid_right)
                    path.append(next_v)
                    used_mask[next_v] = True
        
        return path

    def polish_path(path):
        """Iteratively scan all gaps for insertable vertices."""
        used_mask = np.zeros(MAX_VERTS, dtype=bool)
        for v in path:
            used_mask[v] = True
        
        max_rounds = 200
        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            n = len(path)
            # Scan all gaps: position 0..n (0=prepend, n=append)
            # Go in random order to avoid bias
            positions = list(range(n + 1))
            random.shuffle(positions)
            
            for pos in positions:
                if time.time() - start_time > DEADLINE:
                    break
                
                left_nb = path[pos - 1] if pos > 0 else None
                right_nb = path[pos] if pos < n else None
                
                # Get candidate vertices adjacent to the gap
                candidates = set()
                if left_nb is not None:
                    candidates.update(neighbors[left_nb].tolist())
                if right_nb is not None:
                    candidates.update(neighbors[right_nb].tolist())
                
                for v in candidates:
                    if used_mask[v]:
                        continue
                    if can_insert_fast(used_mask, v, left_nb, right_nb):
                        path.insert(pos, v)
                        used_mask[v] = True
                        n += 1
                        improved = True
                        break  # Restart scan after insertion
                
                if not improved and pos == positions[-1]:
                    break
            
            if not improved:
                break
        
        return path

    # Main loop: multiple restarts with polish
    while time.time() - start_time < DEADLINE:
        # Generate initial path
        initial = greedy_construct()
        
        if len(initial) < 2:
            continue
        
        # Polish it
        polished = polish_path(initial)
        
        if len(polished) > best_len:
            best_len = len(polished)
            best_path = polished
        
        # Try a few more polishes on the best path found so far
        if time.time() - start_time < DEADLINE - 5:
            extra = polish_path(list(best_path))
            if len(extra) > best_len:
                best_len = len(extra)
                best_path = extra
    
    if not best_path:
        best_path = [0, 1]
    
    return best_path
