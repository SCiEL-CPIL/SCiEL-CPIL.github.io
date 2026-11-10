# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

DEADLINE = 630.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    
    Strategy: Use the full unital as initial complement S, then repair by removing
    points that violate the Nikodym condition. This starts with a much larger S (~343)
    than the previous approach (~147), leading to a smaller N and higher score.
    """
    p = 7
    k = 3  # alpha^2 = 3

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)

    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)

    # Enumerate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    n_total = len(all_points)  # 2401

    # Directions: (1, m) for all m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    n_dirs = len(directions)  # 50

    # Precompute line neighbors: line_neighbors[d][i] = list of other points on same line
    line_neighbors = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            if vx == (1, 0):
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:
                c = x
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        line_neighbors.append(neigh)

    # Precompute reverse: for each point j and direction d, which points i share that line with j
    reverse_line = []
    for d_idx in range(n_dirs):
        rev = [[] for _ in range(n_total)]
        for i in range(n_total):
            for j in line_neighbors[d_idx][i]:
                rev[j].append(i)
        reverse_line.append(rev)

    # Base construction: full unital
    S_base = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        if in_unital:
            S_base.add(pt_idx)

    # Start with S = S_base (full unital, ~343 points)
    # We need to remove points from S that violate the Nikodym condition
    # A point i in S violates if for ALL directions d, there exists j != i on line(i,d) with j in S
    
    # Actually, the Nikodym condition says: for every point x (in N or not), there exists a direction d
    # such that all other points on the line through x in direction d are in N.
    # Equivalently: for every point x, there exists d such that NO other point on line(x,d) is in S.
    
    # So a point i in S is "bad" if for all d, there's some j != i on line(i,d) with j in S.
    # We need to remove bad points from S until no bad points remain.
    
    S = set(S_base)
    in_S = np.zeros(n_total, dtype=bool)
    for s in S:
        in_S[s] = True

    # line_s_count[d][i] = number of points in S on line through i in direction d, excluding i
    line_s_count = np.zeros((n_dirs, n_total), dtype=np.int32)
    for d_idx in range(n_dirs):
        for i in range(n_total):
            cnt = 0
            for j in line_neighbors[d_idx][i]:
                if in_S[j]:
                    cnt += 1
            line_s_count[d_idx, i] = cnt

    # clean_dirs_count[i] = number of directions d where line_s_count[d][i] == 0
    clean_dirs_count = np.zeros(n_total, dtype=np.int32)
    for i in range(n_total):
        clean_dirs_count[i] = np.sum(line_s_count[:, i] == 0)

    # A point i is "bad" (must be removed from S) if i in S and clean_dirs_count[i] == 0
    # Remove bad points iteratively
    
    removed = 0
    changed = True
    while changed and _time_left() > 50:
        changed = False
        # Find all bad points
        bad_points = []
        for i in S:
            if clean_dirs_count[i] == 0:
                bad_points.append(i)
        
        if not bad_points:
            break
        
        # Remove one bad point at a time (or all at once for speed)
        # For simplicity, remove all bad points in this round
        for i in bad_points:
            if i not in S:
                continue
            S.remove(i)
            in_S[i] = False
            removed += 1
            changed = True
            
            # Update line_s_count and clean_dirs_count
            for d_idx in range(n_dirs):
                for j in reverse_line[d_idx][i]:
                    if line_s_count[d_idx, j] > 0:
                        old_clean = (line_s_count[d_idx, j] == 0)
                        line_s_count[d_idx, j] -= 1
                        new_clean = (line_s_count[d_idx, j] == 0)
                        if old_clean and not new_clean:
                            clean_dirs_count[j] -= 1
                        elif not old_clean and new_clean:
                            clean_dirs_count[j] += 1

    # Now S is valid: every point has at least one clean direction
    # We can try to add more points to S (remove from N) to increase complement size
    
    # Greedy addition: try to add points to S while maintaining validity
    # A point i can be added to S if after adding, all points still have at least one clean direction
    
    candidates = [i for i in range(n_total) if not in_S[i]]
    rng = np.random.default_rng(42)
    rng.shuffle(candidates)
    
    added = 0
    for cand in candidates:
        if _time_left() < 100:
            break
        
        # Check if cand can be added: it needs at least one clean direction currently
        if clean_dirs_count[cand] == 0:
            continue
        
        # Try adding cand to S
        affected = []
        for d_idx in range(n_dirs):
            for i in reverse_line[d_idx][cand]:
                if line_s_count[d_idx, i] == 0:
                    clean_dirs_count[i] -= 1
                    affected.append((d_idx, i, 0))
                else:
                    affected.append((d_idx, i, line_s_count[d_idx, i]))
                line_s_count[d_idx, i] += 1
        
        # Check if any point lost all clean directions
        valid = True
        for d_idx, i, old_val in affected:
            if clean_dirs_count[i] == 0:
                valid = False
                break
        
        if valid:
            in_S[cand] = True
            S.add(cand)
            added += 1
        else:
            # Rollback
            for d_idx, i, old_val in affected:
                line_s_count[d_idx, i] -= 1
                if line_s_count[d_idx, i] == 0:
                    clean_dirs_count[i] += 1

    # Build final result: N = F_49^2 \ S
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
