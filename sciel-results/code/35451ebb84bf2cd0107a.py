# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a structural change:
    start from the full plane and greedily REMOVE points, using a
    priority queue ordered by how "dangerous" a removal is.
    
    The key structural change: instead of starting from a small excluded set
    and trying to grow it, we start from the full plane (where validity is
    trivially satisfied) and remove points one at a time, always choosing
    the point whose removal is least likely to invalidate other points.
    
    This is a different optimization/search method: greedy deletion with
    priority ordering by impact, rather than constraint propagation + removal.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
    slope_lines = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)
    
    # For each point, list of lines through it
    point_lines = [[] for _ in range(n)]
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        point_lines[i].append(('vert', (xa, xb), None))
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                point_lines[i].append(('slope', (ma, mb), key))
    
    # Start with ALL points in N (trivially valid: every point is on every line through it,
    # and all other points on those lines are in N)
    in_N = [True] * n
    
    # line_not_in_N_count[L] = number of points in L not in N
    line_not_in_N_count = {}
    for key, pts in vert_lines.items():
        line_not_in_N_count[('vert', key)] = 0
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            line_not_in_N_count[('slope', (ma, mb), key)] = 0
    
    # point_good_witness[i] = number of lines L through i where all other points in L are in N
    # Initially all lines have 0 points not in N, so every line is a good witness for every point on it.
    # point_good_witness[i] = number of lines through i = 1 (vertical) + 48 (slope) = 49
    
    point_good_witness = [49] * n
    
    # For greedy removal: we want to remove points that are "safest" to remove.
    # A point i is safe to remove if after removal, no point j loses its last good witness.
    # When we remove i, for each line L through i, line_not_in_N_count[L] increases by 1.
    # For each point j on L (j != i), if L was a good witness for j (i.e., line_not_in_N_count[L] was 0
    # and j was in N, so nin_excl_j = 0), then L is no longer a good witness for j.
    
    # We use a priority queue: points with higher "safety margin" are removed first.
    # Safety margin = minimum over all affected points j of (point_good_witness[j] - 1),
    # where affected means j is on some line L through i and L was a good witness for j.
    # Higher safety margin means more robust.
    
    # Actually, let's think about this differently. We want to maximize the number of points removed.
    # Let's try removing points in a specific order and see how far we get.
    
    # Alternative approach: use the Hermitian unital as a seed for removal.
    # Points on the Hermitian unital are "structurally special" and likely safe to remove.
    
    # Let's combine: start by removing all points on the Hermitian unital (with y_a in QR),
    # then greedily remove more points using the priority queue approach.
    
    qr = set((u*u) % p for u in range(p))  # {0, 1, 2, 4}
    
    # First, remove points on the Hermitian unital with y_a in QR
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
        if in_unital and ya in qr:
            in_N[i] = False
            # Update line_not_in_N_count for all lines through i
            for line_rep in point_lines[i]:
                if line_rep[0] == 'vert':
                    lkey = ('vert', line_rep[1])
                else:
                    lkey = ('slope', line_rep[1], line_rep[2])
                line_not_in_N_count[lkey] += 1
    
    # Recompute point_good_witness
    for i in range(n):
        cnt = 0
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                cnt += 1
        point_good_witness[i] = cnt
    
    # Now greedily remove more points
    # Use a simple approach: repeatedly find a point that can be removed and remove it.
    # To be efficient, we'll iterate through points and try to remove them.
    
    # Sort points by how many good witnesses they have (fewer = more constrained = harder to remove)
    # Actually, we want to remove points that are "easy" to remove, meaning their removal
    # doesn't break too many other points' good witnesses.
    
    # Let's try a different strategy: for each point in N, compute the "cost" of removing it
    # (number of points that would lose their last good witness). Remove the point with cost 0
    # if any, otherwise stop.
    
    max_iter = 1000
    for iteration in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        found_removal = False
        for i in range(n):
            if time.time() > deadline - 30:
                break
            
            if not in_N[i]:
                continue
            
            # Check if i can be removed
            # When we remove i, for each line L through i:
            # - line_not_in_N_count[L] increases by 1
            # - For each point j in L\{i}, if L was a good witness for j, it's no longer
            
            # Check if any point would lose ALL good witnesses
            can_remove = True
            for line_rep in point_lines[i]:
                if line_rep[0] == 'vert':
                    key = line_rep[1]
                    lkey = ('vert', key)
                    line_pts = vert_lines[key]
                else:
                    ma, mb = line_rep[1]
                    key = line_rep[2]
                    lkey = ('slope', (ma, mb), key)
                    line_pts = slope_lines[(ma, mb)][key]
                
                old_count = line_not_in_N_count[lkey]
                new_count = old_count + 1
                
                for j in line_pts:
                    if j == i:
                        continue
                    
                    # Before removal: L is a good witness for j iff old_count - (0 if in_N[j] else 1) == 0
                    # After removal: L is a good witness for j iff new_count - (0 if in_N[j] else 1) == 0
                    
                    was_good = (old_count - (0 if in_N[j] else 1)) == 0
                    still_good = (new_count - (0 if in_N[j] else 1)) == 0
                    
                    if was_good and not still_good:
                        # j loses this good witness
                        # Check if j has other good witnesses
                        if point_good_witness[j] <= 1:
                            can_remove = False
                            break
                
                if not can_remove:
                    break
            
            if can_remove:
                # Remove i
                in_N[i] = False
                found_removal = True
                
                # Update data structures
                for line_rep in point_lines[i]:
                    if line_rep[0] == 'vert':
                        lkey = ('vert', line_rep[1])
                    else:
                        lkey = ('slope', line_rep[1], line_rep[2])
                    line_not_in_N_count[lkey] += 1
                
                # Update point_good_witness for affected points
                for line_rep in point_lines[i]:
                    if line_rep[0] == 'vert':
                        key = line_rep[1]
                        line_pts = vert_lines[key]
                    else:
                        ma, mb = line_rep[1]
                        key = line_rep[2]
                        line_pts = slope_lines[(ma, mb)][key]
                    
                    for j in line_pts:
                        if j == i:
                            continue
                        
                        # Recompute point_good_witness[j]
                        cnt = 0
                        for line_rep_j in point_lines[j]:
                            if line_rep_j[0] == 'vert':
                                lkey_j = ('vert', line_rep_j[1])
                            else:
                                lkey_j = ('slope', line_rep_j[1], line_rep_j[2])
                            
                            total_nin = line_not_in_N_count[lkey_j]
                            nin_excl_j = total_nin - (0 if in_N[j] else 1)
                            if nin_excl_j == 0:
                                cnt += 1
                        point_good_witness[j] = cnt
                break  # Restart the scan after a removal
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
