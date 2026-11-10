# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using line-based complement construction.
    
    Strategy:
    1. Start with S = empty (N = all points).
    2. Greedily add points to S (remove from N) if doing so maintains validity.
    3. Use local validity checks: a point can be removed from N if every point
       that relied on it still has at least one clean witness line.
    4. Prioritize removing points that are "least useful" (fewest witness lines
       depend on them).
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def sub_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
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
    
    # Initialize: N = all points, S = empty
    in_N = [True] * n
    
    # Count how many times each point is "forced" to be in N
    # A point j is forced if there exists a point i such that i's only good witness line L
    # contains j, and all other points on L are in N.
    
    # For each point, track which lines could be its witness
    # and which points on those lines are not in N
    
    # line_not_in_N_count[L] = number of points in L not in N
    line_not_in_N_count = {}
    for key, pts in vert_lines.items():
        line_not_in_N_count[('vert', key)] = sum(1 for j in pts if not in_N[j])
    for (ma, mb), lines in slope_lines.items():
        for key, pts in lines.items():
            line_not_in_N_count[('slope', (ma, mb), key)] = sum(1 for j in pts if not in_N[j])
    
    # point_good_witness[i] = number of lines L through i where all other points in L are in N
    point_good_witness = [0] * n
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
    
    # For each point, track its good witness lines
    point_good_lines = [[] for _ in range(n)]
    for i in range(n):
        good_lines = []
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            
            total_nin = line_not_in_N_count[lkey]
            nin_excl_i = total_nin - (0 if in_N[i] else 1)
            if nin_excl_i == 0:
                good_lines.append(line_rep)
        point_good_lines[i] = good_lines
    
    # Greedily remove points from N (add to S)
    # A point i can be removed if:
    # 1. in_N[i] is True
    # 2. After removal, all points still have at least 1 good witness
    
    # Sort candidates by "usefulness" (ascending) - remove least useful first
    # A point is less useful if fewer other points depend on it as part of their witness lines
    
    # Compute usefulness: for each point i, count how many points j have i on their good witness line
    usefulness = [0] * n
    for i in range(n):
        if not in_N[i]:
            continue
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
                # Check if this line is a good witness for j
                lkey = ('vert', line_rep[1]) if line_rep[0] == 'vert' else ('slope', line_rep[1], line_rep[2])
                total_nin = line_not_in_N_count[lkey]
                nin_excl_j = total_nin - (0 if in_N[j] else 1)
                if nin_excl_j == 0:
                    usefulness[i] += 1
    
    # Try to remove points in order of increasing usefulness
    candidates = [i for i in range(n) if in_N[i]]
    candidates.sort(key=lambda i: usefulness[i])
    
    # Greedy removal with constraint propagation
    removed_count = 0
    for i in candidates:
        if time.time() > deadline - 30:
            break
        
        if not in_N[i]:
            continue
        
        # Check if i has at least 1 good witness
        if len(point_good_lines[i]) == 0:
            continue
        
        # Simulate removal: check if any point would lose all good witnesses
        # When we remove i, for each line L through i:
        # - line_not_in_N_count[L] increases by 1
        # - For each point j in L\{i}, if L was a good witness for j, it's no longer
        
        affected = defaultdict(int)  # j -> number of good witnesses lost
        
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
            
            for j in line_pts:
                if j == i:
                    continue
                
                # Was L a good witness for j?
                was_good = (old_count - (0 if in_N[j] else 1)) == 0
                if was_good:
                    affected[j] += 1
        
        # Check if any affected point would have 0 good witnesses
        invalid = False
        for j, lost in affected.items():
            if len(point_good_lines[j]) - lost <= 0:
                invalid = True
                break
        
        if invalid:
            continue
        
        # Remove i
        in_N[i] = False
        removed_count += 1
        
        # Update data structures
        for line_rep in point_lines[i]:
            if line_rep[0] == 'vert':
                lkey = ('vert', line_rep[1])
            else:
                lkey = ('slope', line_rep[1], line_rep[2])
            line_not_in_N_count[lkey] += 1
        
        # Update point_good_lines for affected points
        for j in affected.keys():
            new_good_lines = []
            for line_rep_j in point_lines[j]:
                if line_rep_j[0] == 'vert':
                    lkey = ('vert', line_rep_j[1])
                else:
                    lkey = ('slope', line_rep_j[1], line_rep_j[2])
                
                total_nin = line_not_in_N_count[lkey]
                nin_excl_j = total_nin - (0 if in_N[j] else 1)
                if nin_excl_j == 0:
                    new_good_lines.append(line_rep_j)
            point_good_lines[j] = new_good_lines
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
