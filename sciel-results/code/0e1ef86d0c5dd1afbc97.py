# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_LINES_TOTAL = 2450
N_PER_LINE = p*p  # 49

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    """
    Precompute all 2450 lines.
    Returns a numpy array of shape (N_LINES_TOTAL, N_PER_LINE) with point indices.
    """
    lines = np.zeros((N_LINES_TOTAL, N_PER_LINE), dtype=np.int32)
    
    line_idx = 0
    # Direction 0: v = (0, 1), vertical
    for xa in range(p):
        for xb in range(p):
            for k in range(N_PER_LINE):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    for k in range(N_PER_LINE):
                        xa = k // p
                        xb = k % p
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    
    return lines

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precompute lines and point_lines
    lines = _precompute_lines()
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(N_LINES_TOTAL):
        for idx in lines[i]:
            point_lines[idx].append(i)
    
    best_S_size = 0
    best_in_S = None
    
    # Use multiple random restarts
    # Each restart:
    # 1. Init S (seed or empty or random small set)
    # 2. Greedy add
    # 3. Update best
    
    # We will try a few different initialization strategies
    # Strategy 1: Seed
    # Strategy 2: Empty
    # Strategy 3: Random subset
    
    strategies = [
        "seed",
        "empty",
        "random_100",
        "random_500"
    ]
    
    rng = np.random.default_rng(42)
    
    for strat in strategies:
        if time.time() > _DEADLINE - 10:
            break
            
        if strat == "seed":
            in_S = _generate_seed_excluded().astype(np.int8)
        elif strat == "empty":
            in_S = np.zeros(N_TOTAL, dtype=np.int8)
        elif strat == "random_100":
            in_S = np.zeros(N_TOTAL, dtype=np.int8)
            indices = rng.choice(N_TOTAL, 100, replace=False)
            in_S[indices] = 1
        elif strat == "random_500":
            in_S = np.zeros(N_TOTAL, dtype=np.int8)
            indices = rng.choice(N_TOTAL, 500, replace=False)
            in_S[indices] = 1
        else:
            continue
            
        # Compute initial line_s_counts and bad
        line_s_counts = np.sum(in_S[lines], axis=1)
        bad = np.zeros(N_TOTAL, dtype=np.int32)
        
        for i in range(N_LINES_TOTAL):
            cnt = line_s_counts[i]
            if cnt > 0:
                line_pts = lines[i, :]
                mask = cnt > in_S[line_pts]
                bad[line_pts[mask]] += 1
        
        # Check if initial state is valid (max bad < 50)
        if np.max(bad) >= 50:
            # If invalid, we need to remove points to fix it?
            # Greedy addition assumes we start valid.
            # If we start with random 500, it's likely invalid.
            # We can try to "repair" by removing points that are most "bad".
            # Or just skip this strategy if it starts invalid and we don't have a repair algorithm.
            # Let's implement a simple repair: while invalid, remove point with highest bad contribution?
            # Or just skip to next strategy if invalid.
            # Actually, let's try to repair by removing points.
            # If bad[x] == 50, x is invalid.
            # We must remove some points from S to reduce bad[x].
            # This is complex. Let's stick to strategies that start valid or are easily made valid.
            # Empty and Seed are valid.
            # Random 100/500 might be invalid.
            # If invalid, we can try to remove random points until valid?
            # Or just skip.
            # Let's skip invalid starts to ensure correctness and simplicity.
            # The greedy addition from a valid state is the core.
            pass
        else:
            # Greedy Addition Loop
            # We do multiple passes with random shuffling
            progress = True
            while progress:
                if time.time() > _DEADLINE - 5:
                    break
                progress = False
                candidates = [i for i in range(N_TOTAL) if not in_S[i]]
                rng.shuffle(candidates)
                
                for c_idx in candidates:
                    if time.time() > _DEADLINE - 2:
                        break
                    can_add = True
                    for l_idx in point_lines[c_idx]:
                        cnt = line_s_counts[l_idx]
                        if cnt == 0:
                            # Affected: x in L, x != c, x not in S
                            line_pts = lines[l_idx, :]
                            for x in line_pts:
                                if x != c_idx and not in_S[x]:
                                    if bad[x] >= 49:
                                        can_add = False
                                        break
                            if not can_add:
                                break
                        elif cnt == 1:
                            # Affected: x in L, x != c, x in S
                            line_pts = lines[l_idx, :]
                            for x in line_pts:
                                if x != c_idx and in_S[x]:
                                    if bad[x] >= 49:
                                        can_add = False
                                        break
                            if not can_add:
                                break
                        
                    if can_add:
                        in_S[c_idx] = 1
                        progress = True
                        for l_idx in point_lines[c_idx]:
                            cnt = line_s_counts[l_idx]
                            line_s_counts[l_idx] += 1
                            line_pts = lines[l_idx, :]
                            if cnt == 0:
                                for x in line_pts:
                                    if x != c_idx and not in_S[x]:
                                        bad[x] += 1
                            elif cnt == 1:
                                for x in line_pts:
                                    if x != c_idx and in_S[x]:
                                        bad[x] += 1
                                        break # Only one such point (the one in S)
            
            current_size = np.sum(in_S)
            if current_size > best_S_size:
                best_S_size = current_size
                best_in_S = in_S.copy()
    
    # If best_in_S is None (e.g. all strategies failed or were invalid), fallback to seed
    if best_in_S is None:
        best_in_S = _generate_seed_excluded().astype(np.int8)
        
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not best_in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
