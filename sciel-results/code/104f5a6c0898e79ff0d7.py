import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0

p = 7
N_TOTAL = 2401
N_DIRS = 50

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _precompute_neighbors():
    """
    Returns a list of lists.
    neighbors[y][v] is a list of indices x such that x is on the line through y with direction v, x != y.
    """
    neighbors = [[[] for _ in range(N_DIRS)] for _ in range(N_TOTAL)]
    
    # Direction 0: v = (0, 1)
    # Line through (xa, xb, ya, yb) has constant x.
    # Points are (xa, xb, ya', yb') for all ya', yb'.
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    y_idx = _point_to_idx(xa, xb, ya, yb)
                    for ya2 in range(p):
                        for yb2 in range(p):
                            if ya2 == ya and yb2 == yb:
                                continue
                            x_idx = _point_to_idx(xa, xb, ya2, yb2)
                            neighbors[y_idx][0].append(x_idx)
    
    # Directions 1..49: v = (1, m)
    for ma in range(p):
        for mb in range(p):
            v_idx = 1 + ma * p + mb
            # Iterate over all points y
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            y_idx = _point_to_idx(xa, xb, ya, yb)
                            # t in F_49, t != 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # Calculate x'
                                    xa2 = (xa + ta) % p
                                    xb2 = (xb + tb) % p
                                    # Calculate y'
                                    # ya' = ya + ta*ma + 3*tb*mb
                                    # yb' = yb + ta*mb + tb*ma
                                    ya2 = (ya + ta * ma + 3 * tb * mb) % p
                                    yb2 = (yb + ta * mb + tb * ma) % p
                                    x_idx = _point_to_idx(xa2, xb2, ya2, yb2)
                                    neighbors[y_idx][v_idx].append(x_idx)
                                    
    return neighbors

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    neighbors = _precompute_neighbors()
    
    best_S_size = 0
    best_S = np.zeros(N_TOTAL, dtype=np.int8)
    
    # Run multiple random restarts
    while time.time() < _DEADLINE - 10:
        # Initialize state
        in_S = np.zeros(N_TOTAL, dtype=np.int8)
        valid = np.ones((N_TOTAL, N_DIRS), dtype=np.int8)
        valid_count = np.full(N_TOTAL, N_DIRS, dtype=np.int32)
        
        candidates = list(range(N_TOTAL))
        np.random.shuffle(candidates)
        
        added = 0
        for y in candidates:
            if in_S[y]:
                continue
            
            # Check if adding y is safe
            safe = True
            # We need to track changes to rollback if not safe
            # But since we discard y if not safe, we can just check
            # Actually, we need to know if valid_count[x] drops to 0.
            
            # Optimization: Pre-compute the list of (x, v) pairs that would be affected
            # affected_pairs = []
            # for v in range(N_DIRS):
            #    for x in neighbors[y][v]:
            #       if valid[x][v]:
            #          affected_pairs.append((x, v))
            
            # Check conflicts
            # We can do this in two passes or one pass with rollback.
            # One pass with rollback is safer.
            
            changes = []
            for v in range(N_DIRS):
                for x in neighbors[y][v]:
                    if valid[x][v]:
                        if valid_count[x] == 1:
                            safe = False
                            break
                        else:
                            # Record change
                            changes.append((x, v))
                            valid_count[x] -= 1
                            valid[x][v] = 0
                if not safe:
                    break
            
            if safe:
                in_S[y] = 1
                added += 1
            else:
                # Rollback
                for x, v in changes:
                    valid_count[x] += 1
                    valid[x][v] = 1
                # Discard y (it will never be added in this run)
        
        if added > best_S_size:
            best_S_size = added
            best_S = in_S.copy()
            
        # Check time
        if time.time() > _DEADLINE - 5:
            break
            
    # Build result N = complement of S
    # N contains points not in S
    points = []
    for idx in range(N_TOTAL):
        if not best_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
