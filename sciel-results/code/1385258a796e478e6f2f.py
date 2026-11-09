import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3
    
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def get_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Precompute lines
    directions = []
    directions.append((0, 0, 1, 0))
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
            
    all_lines = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx_start = get_idx(xa, xb, ya, yb)
                    for d in directions:
                        dxa, dxb, dya, dyb = d
                        line_points = []
                        for t in range(p):
                            cxa = (xa + t * dxa) % p
                            cxb = (xb + t * dxb) % p
                            cya = (ya + t * dya) % p
                            cyb = (yb + t * dyb) % p
                            line_points.append(get_idx(cxa, cxb, cya, cyb))
                        all_lines.append(line_points)
                        
    point_to_lines = {i: [] for i in range(q*q)}
    for i, line in enumerate(all_lines):
        for idx in line:
            point_to_lines[idx].append(i)
            
    # Baseline S
    S_base = set()
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S_base.add(get_idx(xa, xb, ya, yb))
                        
    S = set(S_base)
    
    # Initialize disjoint line counts for points in N
    # disjoint_count[x] = number of lines through x disjoint from S
    disjoint_count = np.zeros(q*q, dtype=int)
    
    for x in range(q*q):
        if x in S:
            continue
        count = 0
        for line_idx in point_to_lines[x]:
            line = all_lines[line_idx]
            disjoint = True
            for y in line:
                if y in S:
                    disjoint = False
                    break
            if disjoint:
                count += 1
        disjoint_count[x] = count
        
    # Check if any point in N has 0 disjoint lines
    # If so, baseline is invalid (should not happen)
    
    # Greedy expansion
    start_time = time.time()
    timeout = 500
    
    candidates = [i for i in range(q*q) if i not in S]
    np.random.shuffle(candidates)
    
    for p_idx in candidates:
        if time.time() - start_time > timeout:
            break
            
        # Check if p_idx can be added
        # 1. p_idx must have a line disjoint from current S
        # This is equivalent to disjoint_count_old[p_idx] > 0?
        # No, disjoint_count is for points in N. p_idx is in N.
        # So we check if disjoint_count[p_idx] > 0.
        if disjoint_count[p_idx] == 0:
            continue
            
        # 2. Adding p_idx to S will reduce disjoint_count for points on lines through p_idx.
        # For each line L through p_idx:
        #    For each x in L \ {p_idx}:
        #       If L was disjoint from S, it is now not disjoint.
        #       So disjoint_count[x] -= 1.
        #       If disjoint_count[x] becomes 0, x is blocked.
        
        # We need to check if any x becomes 0.
        # We can simulate this.
        
        new_counts = disjoint_count.copy()
        valid = True
        
        for line_idx in point_to_lines[p_idx]:
            line = all_lines[line_idx]
            # Check if this line was disjoint from S
            # If it was, then for all x in line \ {p_idx}, the count decreases.
            # But we need to know if it WAS disjoint.
            # We can check: if any y in line (y != p_idx) is in S, then it was NOT disjoint.
            # Actually, if the line was disjoint from S, then no y in line is in S.
            # Since p_idx is not in S, if the line is disjoint from S, then no y in line is in S.
            # So we can just check if the line contains any point from S.
            
            is_disjoint_from_S = True
            for y in line:
                if y in S:
                    is_disjoint_from_S = False
                    break
                    
            if is_disjoint_from_S:
                for x in line:
                    if x == p_idx:
                        continue
                    new_counts[x] -= 1
                    if new_counts[x] < 0:
                         # Should not happen if logic is correct
                         pass
                    if new_counts[x] == 0:
                         # x is blocked
                         valid = False
                         break
            if not valid:
                break
                
        if valid:
            # Also need to check p_idx itself?
            # p_idx is added to S.
            # For p_idx, we need a line L such that L \cap S_new \subseteq {p_idx}.
            # This is the same as L \cap S \subseteq {p_idx} \setminus \{p_idx\} = empty?
            # No, L \cap S_new \subseteq {p_idx} means L \cap S \subseteq {p_idx} is not quite right.
            # It means no other point in L is in S_new.
            # Since p_idx is the only new point, this means no other point in L is in S.
            # So L must be disjoint from S.
            # We already checked disjoint_count[p_idx] > 0, which means there is a line through p_idx disjoint from S.
            # So Condition 1 is satisfied.
            
            # Update S and disjoint_count
            S.add(p_idx)
            disjoint_count = new_counts
        else:
            # Cannot add
            pass

    # Construct output
    points = []
    for i in range(q*q):
        if i not in S:
            xa, xb, ya, yb = get_coords(i)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
