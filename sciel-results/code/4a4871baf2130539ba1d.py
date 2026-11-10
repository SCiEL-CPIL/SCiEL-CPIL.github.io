# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using simulated annealing on the complement set S.
    
    The Nikodym property: for every point x in F_49^2, there exists a direction d such
    that the line L(x,d) contains no points of S other than possibly x.
    
    We maximize |S| (the complement of N). The energy is the number of "blocked" points
    (points with no safe direction). We use simulated annealing with add/remove/swap moves.
    """
    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    num_dirs = 50
    
    # Precompute lines
    direction_lines = []
    
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    for ma in range(p):
        for mb in range(p):
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * num_dirs for _ in range(n)]
    for dir_idx in range(num_dirs):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # For each point and direction, list of other points on the same line
    line_members = [[[] for _ in range(num_dirs)] for _ in range(n)]
    for d in range(num_dirs):
        for line in direction_lines[d]:
            for i in range(len(line)):
                for j in range(len(line)):
                    if i != j:
                        line_members[line[i]][d].append(line[j])
    
    # Precompute: for each point p, list of (q, d) pairs where q is on the same line as p in direction d
    # This is the "impact" of adding p to S
    impact = [[] for _ in range(n)]
    for p_idx in range(n):
        for d in range(num_dirs):
            li = point_line[p_idx][d]
            line = direction_lines[d][li]
            for q in line:
                if q != p_idx:
                    impact[p_idx].append((q, d))
    
    def compute_blocked_count(in_S):
        """Count number of points with no safe direction."""
        blocked = 0
        for x in range(n):
            has_safe = False
            for d in range(num_dirs):
                clean = True
                for q in line_members[x][d]:
                    if in_S[q]:
                        clean = False
                        break
                if clean:
                    has_safe = True
                    break
            if not has_safe:
                blocked += 1
        return blocked
    
    def greedy_initial():
        """Greedy construction of a valid S."""
        in_S = [False] * n
        safe_mask = [(1 << num_dirs) - 1] * n
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            valid = True
            changes = []
            
            for d in range(num_dirs):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if (safe_mask[q] >> d) & 1:
                        if safe_mask[q] == (1 << d):
                            valid = False
                            break
                        else:
                            changes.append((q, d))
                            safe_mask[q] &= ~(1 << d)
                if not valid:
                    break
            
            if valid:
                in_S[idx] = True
            else:
                for q, d in changes:
                    safe_mask[q] |= (1 << d)
        
        return in_S
    
    best_S = None
    best_size = 0
    
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        in_S = greedy_initial()
        current_size = sum(in_S)
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
        
        # Simulated annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        max_iters = 50000
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            if T < T_min:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random point not in S
                candidates = [i for i in range(n) if not in_S[i]]
                if not candidates:
                    continue
                pt = random.choice(candidates)
                
                # Check if adding pt would block any point
                delta_blocked = 0
                for q, d in impact[pt]:
                    if in_S[q]:
                        continue
                    # q is not in S, check if d is its only safe direction
                    only_safe = True
                    for d2 in range(num_dirs):
                        if d2 == d:
                            continue
                        clean = True
                        for q2 in line_members[q][d2]:
                            if in_S[q2]:
                                clean = False
                                break
                        if clean:
                            only_safe = False
                            break
                    if only_safe:
                        delta_blocked += 1
                
                if delta_blocked == 0:
                    in_S[pt] = True
                    current_size += 1
                else:
                    # Accept with probability exp(-delta_blocked/T)
                    if random.random() < np.exp(-delta_blocked / T):
                        in_S[pt] = True
                        current_size += 1
            elif move_type < 0.7:
                # Try to remove a random point from S
                candidates = [i for i in range(n) if in_S[i]]
                if not candidates:
                    continue
                pt = random.choice(candidates)
                in_S[pt] = False
                current_size -= 1
            else:
                # Swap: remove one from S, add one not in S
                in_candidates = [i for i in range(n) if in_S[i]]
                out_candidates = [i for i in range(n) if not in_S[i]]
                if not in_candidates or not out_candidates:
                    continue
                pt_out = random.choice(in_candidates)
                pt_in = random.choice(out_candidates)
                
                # Remove pt_out
                in_S[pt_out] = False
                # Add pt_in - check blocking
                delta_blocked = 0
                for q, d in impact[pt_in]:
                    if in_S[q]:
                        continue
                    only_safe = True
                    for d2 in range(num_dirs):
                        if d2 == d:
                            continue
                        clean = True
                        for q2 in line_members[q][d2]:
                            if in_S[q2]:
                                clean = False
                                break
                        if clean:
                            only_safe = False
                            break
                    if only_safe:
                        delta_blocked += 1
                
                if delta_blocked == 0:
                    in_S[pt_in] = True
                else:
                    if random.random() < np.exp(-delta_blocked / T):
                        in_S[pt_in] = True
                    else:
                        # Revert: put pt_out back
                        in_S[pt_out] = True
                        current_size += 1
                        T *= cooling
                        continue
            
            T *= cooling
        
        # Final validation and update best
        if sum(in_S) > best_size:
            # Validate
            blocked = compute_blocked_count(in_S)
            if blocked == 0:
                best_size = sum(in_S)
                best_S = in_S[:]
    
    # Fallback: if no valid set found, use empty S (N = all points)
    if best_S is None:
        best_S = [False] * n
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
