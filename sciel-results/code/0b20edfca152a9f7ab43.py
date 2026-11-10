# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute all lines in F_49^2.
    2. Start with S = empty.
    3. Greedily add points to S (complement of N) if it maintains the Nikodym condition.
       Condition: For every point x in F_49^2, there exists a line L through x such that L \ {x} is disjoint from S.
       (If x in S, L \ {x} disjoint from S means L \cap S = {x}.
        If x not in S, L \ {x} disjoint from S means L \cap S = empty).
       
       We want to maximize |S|.
       
       Incremental update logic:
       Let C[x] be the number of "valid" lines through x.
       A line L is valid for x if:
       - x in S: L \cap S = {x}
       - x not in S: L \cap S = empty
       
       When adding a new point x to S:
       - For each line L through x:
         Let k = |L \cap S_old|.
         If k == 0:
           L was valid for all y in L (since y not in S_old, L \cap S_old = empty).
           After adding x:
           - For x: L \cap S_new = {x}. Valid for x.
           - For y in L \ {x}: L \cap S_new = {x}. Since y not in S_new, L is NOT valid for y (needs empty intersection).
             So C[y] decreases by 1 for all y in L \ {x}.
         If k == 1:
           Let y0 be the unique point in L \cap S_old.
           L was valid for y0 (L \cap S_old = {y0}).
           After adding x:
           - For y0: L \cap S_new = {y0, x}. Not valid for y0. C[y0] decreases by 1.
           - For x: L \cap S_new = {y0, x}. Not valid for x.
           - For others: No change in validity status relative to S_old (they were not valid before, and remain not valid).
             Wait, if z not in S_old and z in L:
             L \cap S_old had y0. So L was not valid for z.
             L \cap S_new has y0, x. Still not valid for z.
           So only C[y0] decreases.
         If k >= 2:
           L was not valid for anyone in S_old (intersection size > 1).
           L was not valid for anyone not in S_old (intersection size > 0).
           After adding x, intersection size >= 3. Still not valid for anyone.
           No changes to C.
       
       We can add x if:
       1. There is at least one line L through x with k == 0. (So x has a valid witness in S_new).
       2. No point y has its C[y] drop to 0.
       
       This allows us to maintain the state efficiently.
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin

    p = 7
    q = p * p
    k_const = 3  # alpha^2 = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    num_points = q * q
    
    # Precompute lines
    lines = []
    point_to_lines = [[] for _ in range(num_points)]
    
    seen_lines = set()
    
    # Directions: (1, m) for m in F_49, and (0, 1).
    
    # For (1, m): y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)
                    
                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        line_id = len(lines) - 1
                        for idx in line_points:
                            point_to_lines[idx].append(line_id)
    
    # For (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            c = (ca, cb)
            line_points = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((ca * p + cb) * p + ya) * p + yb
                    line_points.append(idx)
                    
            line_tuple = tuple(sorted(line_points))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                line_set = set(line_points)
                lines.append(line_set)
                line_id = len(lines) - 1
                for idx in line_points:
                    point_to_lines[idx].append(line_id)

    # State
    S = set()
    # C[x] = number of valid lines for x
    C = [0] * num_points
    
    # Initialize C for S = empty
    # All lines are disjoint from S, so all lines through x are valid.
    for x in range(num_points):
        C[x] = len(point_to_lines[x])
        
    candidates = set(range(num_points))
    
    while candidates:
        if time.time() > deadline:
            break
            
        # Pick candidate with smallest C[x] > 0
        best_x = None
        best_count = float('inf')
        # Optimization: Iterate over a sorted list or just scan.
        # Scanning 2401 items is fast.
        for x in candidates:
            if C[x] > 0 and C[x] < best_count:
                best_count = C[x]
                best_x = x
                
        if best_x is None:
            break
            
        x = best_x
        
        # Check if adding x is valid
        valid = True
        
        # We need to simulate the changes to C.
        # Instead of simulating and rolling back, we can just check if any C[y] would become 0.
        
        # Collect points that would be affected and how much their C decreases.
        # delta_C = { y: -1, ... }
        
        # We iterate over lines through x.
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            
            # Count |L \cap S|
            # We can do this quickly if we maintain counts?
            # Or just iterate L (size 49).
            
            s_count = 0
            s_point = None
            for y in L:
                if y in S:
                    s_count += 1
                    s_point = y
                    if s_count > 1:
                        break
            
            if s_count == 0:
                # L \cap S = empty
                # For all y in L \ {x}, C[y] -= 1
                for y in L:
                    if y == x:
                        continue
                    if C[y] == 1:
                        valid = False
                        break
                if not valid:
                    break
                    
            elif s_count == 1:
                # L \cap S = {s_point}
                # C[s_point] -= 1
                if C[s_point] == 1:
                    valid = False
                    break
            else:
                # No change
                pass
        
        if not valid:
            candidates.discard(x)
            continue
            
        # Commit
        S.add(x)
        candidates.discard(x)
        
        # Update C
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            
            s_count = 0
            s_point = None
            # Note: S now contains x.
            # We need to know the state BEFORE adding x to determine the transition.
            # Let's re-calculate based on S \ {x}.
            
            # Actually, we can just re-calculate the intersection with S_old = S \ {x}.
            # Since we just added x, S_old = S - {x}.
            
            s_old_count = 0
            s_old_point = None
            for y in L:
                if y in S and y != x:
                    s_old_count += 1
                    s_old_point = y
                    if s_old_count > 1:
                        break
            
            if s_old_count == 0:
                # Transition: Empty -> {x}
                # For all y in L \ {x}, C[y] -= 1
                for y in L:
                    if y != x:
                        C[y] -= 1
            elif s_old_count == 1:
                # Transition: {y0} -> {y0, x}
                # C[y0] -= 1
                C[s_old_point] -= 1
            else:
                # No change
                pass
                
        # Update C[x]
        # x is in S. Valid lines for x are those with L \cap S = {x}.
        # i.e., L \cap (S \ {x}) = empty.
        count_x = 0
        for line_id in point_to_lines[x]:
            L = lines[line_id]
            has_other = False
            for y in L:
                if y != x and y in S:
                    has_other = True
                    break
            if not has_other:
                count_x += 1
        C[x] = count_x

    # Construct N = F^2 \ S
    N = [i for i in range(num_points) if i not in S]
    
    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
