# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search with pair swaps.
    
    Strategy:
    1. Greedy initialization to get a valid S (complement of N).
    2. Local search: repeatedly try to (a) add a single point to S, or 
       (b) perform a swap: remove one point from S and add two new points.
       This is a fundamentally different search mechanism than pure greedy.
    """
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Precompute affected_pairs: for each point p, list of (q, d) where q != p and p is on line L(q, d)
    affected_pairs = [[] for _ in range(n)]
    for d in range(50):
        for line in direction_lines[d]:
            for i, q in enumerate(line):
                for j, r in enumerate(line):
                    if i != j:
                        affected_pairs[r].append((q, d))
    
    # Helper: build safe_mask from scratch given in_S
    def build_safe_mask(in_S):
        safe_mask = [(1 << 50) - 1] * n
        for idx in range(n):
            if in_S[idx]:
                for q, d in affected_pairs[idx]:
                    safe_mask[q] &= ~(1 << d)
        return safe_mask
    
    # Helper: check if a set S is valid (no point has empty safe_mask)
    def is_valid_S(in_S, safe_mask):
        for i in range(n):
            if safe_mask[i] == 0:
                return False
        return True
    
    # Helper: try to add point idx to S, return (success, new_safe_mask, changes)
    def try_add_point(in_S, safe_mask, idx):
        if in_S[idx]:
            return False, safe_mask, []
        
        valid = True
        changes = []
        
        for q, d in affected_pairs[idx]:
            if (safe_mask[q] >> d) & 1:
                if safe_mask[q] == (1 << d):
                    valid = False
                    break
                else:
                    changes.append((q, d))
                    safe_mask[q] &= ~(1 << d)
        
        if valid:
            in_S[idx] = True
            return True, safe_mask, changes
        else:
            # Rollback
            for q, d in changes:
                safe_mask[q] |= (1 << d)
            return False, safe_mask, []
    
    # Helper: try to remove point idx from S, return new_safe_mask
    def try_remove_point(in_S, safe_mask, idx):
        if not in_S[idx]:
            return safe_mask
        in_S[idx] = False
        for q, d in affected_pairs[idx]:
            safe_mask[q] |= (1 << d)
        return safe_mask
    
    # Phase 1: Multiple random restarts with greedy construction
    best_S = [False] * n
    best_size = 0
    best_safe_mask = None
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 120:
            break
        
        # Randomize order
        order = list(range(n))
        random.shuffle(order)
        
        in_S = [False] * n
        safe_mask = [(1 << 50) - 1] * n
        current_size = 0
        
        for idx in order:
            success, safe_mask, _ = try_add_point(in_S, safe_mask, idx)
            if success:
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_S = in_S[:]
            best_safe_mask = safe_mask[:]
    
    # Phase 2: Local search - single additions and pair swaps
    if best_safe_mask is not None:
        in_S = best_S[:]
        safe_mask = best_safe_mask[:]
        current_size = best_size
        no_improve_count = 0
        
        while time.time() < deadline - 60:
            if no_improve_count > 100:
                break
            
            improved = False
            
            # Try single additions
            candidates = [i for i in range(n) if not in_S[i]]
            random.shuffle(candidates)
            
            for idx in candidates:
                if time.time() > deadline - 60:
                    break
                success, safe_mask, _ = try_add_point(in_S, safe_mask, idx)
                if success:
                    current_size += 1
                    improved = True
                    no_improve_count = 0
                    break
            
            # Try pair swaps: remove one from S, add two not in S
            if not improved:
                s_candidates = [i for i in range(n) if in_S[i]]
                if s_candidates:
                    random.shuffle(s_candidates)
                    
                    for rem_idx in s_candidates[:50]:  # Limit tries
                        if time.time() > deadline - 60:
                            break
                        
                        # Temporarily remove rem_idx
                        old_safe = safe_mask[:]
                        safe_mask = try_remove_point(in_S, safe_mask, rem_idx)
                        
                        # Try to add two points
                        add_candidates = [i for i in range(n) if not in_S[i] and i != rem_idx]
                        random.shuffle(add_candidates)
                        
                        added = []
                        for add_idx in add_candidates[:100]:
                            if len(added) == 2:
                                break
                            success, safe_mask, _ = try_add_point(in_S, safe_mask, add_idx)
                            if success:
                                added.append(add_idx)
                        
                        if len(added) >= 2:
                            current_size += len(added) - 1
                            improved = True
                            no_improve_count = 0
                        else:
                            # Rollback
                            in_S[rem_idx] = True
                            safe_mask = old_safe
                            for add_idx in added:
                                in_S[add_idx] = False
                                for q, d in affected_pairs[add_idx]:
                                    safe_mask[q] |= (1 << d)
            
            if not improved:
                no_improve_count += 1
    
    # Build result: N = All \ S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
