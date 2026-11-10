# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3
    total_points = q * q  # 2401
    
    deadline = time.time() + 580  # safety margin
    
    # Field operations
    def add_f49(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)
    
    def mul_f49(u, v):
        a, b = u
        c, d = v
        return ((a*c + b*d*k) % p, (a*d + b*c) % p)
    
    # Generate all points
    points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa, xb), (ya, yb))
                    idx = len(points)
                    points.append(pt)
                    point_to_idx[pt] = idx
    
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) vertical = 50
    directions = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    
    # Precompute lines: for each point, list of line_ids
    # Each line identified by (point, direction) but we need unique lines
    line_map = {}
    lines = []  # list of sets of point indices
    point_to_lines = [[] for _ in range(total_points)]
    
    for x_idx in range(total_points):
        x = points[x_idx]
        for dx, dy in directions:
            line_pts = set()
            for ta in range(p):
                for tb in range(p):
                    t = (ta, tb)
                    px = add_f49(x[0], mul_f49(t, dx))
                    py = add_f49(x[1], mul_f49(t, dy))
                    pt = (px, py)
                    line_pts.add(point_to_idx[pt])
            line_key = frozenset(line_pts)
            if line_key not in line_map:
                line_map[line_key] = len(lines)
                lines.append(line_pts)
            lid = line_map[line_key]
            point_to_lines[x_idx].append(lid)
    
    num_lines = len(lines)
    
    # Line counts: how many S-points on each line
    # For validity: 
    #   x in N: exists line L through x with count(L) == 0
    #   x in S: exists line L through x with count(L) == 1
    
    # Start with S = empty, then greedily add points
    # Better: start with a good initial S and use simulated annealing
    
    # Initial S: use the unital-based construction
    S = set()
    for pt, idx in point_to_idx.items():
        xa, xb = pt[0]
        ya, yb = pt[1]
        if (2 * xa + ya * ya - yb * yb * k) % p == 0:
            if ya in {0, 3, 5, 6}:
                S.add(idx)
    
    line_counts = np.zeros(num_lines, dtype=np.int32)
    for idx in S:
        for lid in point_to_lines[idx]:
            line_counts[lid] += 1
    
    def is_valid(S_set, lc):
        for x_idx in range(total_points):
            if x_idx in S_set:
                found = False
                for lid in point_to_lines[x_idx]:
                    if lc[lid] == 1:
                        found = True
                        break
                if not found:
                    return False
            else:
                found = False
                for lid in point_to_lines[x_idx]:
                    if lc[lid] == 0:
                        found = True
                        break
                if not found:
                    return False
        return True
    
    # Verify initial S
    if not is_valid(S, line_counts):
        # Fallback: empty S is always valid
        S = set()
        line_counts = np.zeros(num_lines, dtype=np.int32)
    
    best_S = set(S)
    best_size = len(S)
    
    # Simulated annealing
    random.seed(42)
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute candidate points (all points not in S initially, plus S points for swaps)
    all_indices = list(range(total_points))
    
    step = 0
    while time.time() < deadline and T > min_T:
        step += 1
        # Pick a random point to add or remove
        if random.random() < 0.6:
            # Try adding a point
            if len(S) >= total_points - 1:
                T *= cooling
                continue
            # Pick random point not in S
            idx = random.choice(all_indices)
            if idx in S:
                continue
            # Add it
            S.add(idx)
            for lid in point_to_lines[idx]:
                line_counts[lid] += 1
            if is_valid(S, line_counts):
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            else:
                # Revert
                S.remove(idx)
                for lid in point_to_lines[idx]:
                    line_counts[lid] -= 1
        else:
            # Try removing a point (to allow restructuring)
            if len(S) == 0:
                T *= cooling
                continue
            idx = random.choice(list(S))
            S.remove(idx)
            for lid in point_to_lines[idx]:
                line_counts[lid] -= 1
            # After removal, S is smaller but still valid (removing points preserves validity)
        
        T *= cooling
    
    # Use best_S found
    S = best_S
    
    # Construct N = complement of S
    N_points = []
    for pt, idx in point_to_idx.items():
        if idx not in S:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
    
    if not N_points:
        # Should not happen, but safety
        N_points = [[[0, 0], [0, 0]]]
    
    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
