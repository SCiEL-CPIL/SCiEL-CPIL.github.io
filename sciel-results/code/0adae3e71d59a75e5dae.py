import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses greedy construction of the excluded set C with multiple random restarts.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 600s kill
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n_points = len(all_points)
    
    # Precompute lines
    # Vertical lines: x = const (v = (0, 1))
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) if x_a == xa and x_b == xb]
            vert_lines.append(pts)
            
    # Slope lines: y = m*x + c, v = (1, m)
    slope_lines = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            for c, pts in lines_dict.items():
                slope_lines.append(pts)
                
    # Combine all lines
    all_lines = vert_lines + slope_lines
    n_lines = len(all_lines)
        
    # Map point to lines
    point_to_lines = [[] for _ in range(n_points)]
    for idx, line in enumerate(all_lines):
        for pt_idx in line:
            point_to_lines[pt_idx].append(idx)
            
    def try_greedy(in_C, n_free, n_iso, deadline):
        """
        Run greedy expansion from current state.
        Returns (in_C, n_free, n_iso) after expansion.
        """
        improved = True
        while improved:
            improved = False
            for x_idx in range(n_points):
                if in_C[x_idx]:
                    continue
                if time.time() > deadline - 5:
                    return in_C, n_free, n_iso
                    
                # Check if x_idx can be added
                if n_free[x_idx] < 1:
                    continue
                    
                valid = True
                lines_to_update = []
                
                for line_idx in point_to_lines[x_idx]:
                    line = all_lines[line_idx]
                    c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
                    
                    if len(c_pts) == 0:
                        for y_idx in line:
                            if y_idx == x_idx:
                                continue
                            if n_free[y_idx] < 2:
                                valid = False
                                break
                        if not valid:
                            break
                        lines_to_update.append((line_idx, 'empty'))
                    elif len(c_pts) == 1:
                        y0 = c_pts[0]
                        if n_iso[y0] < 2:
                            valid = False
                            break
                        lines_to_update.append((line_idx, 'single', y0))
                    else:
                        pass
                        
                if valid:
                    in_C[x_idx] = True
                    improved = True
                    
                    for update in lines_to_update:
                        if update[1] == 'empty':
                            line_idx = update[0]
                            line = all_lines[line_idx]
                            for y_idx in line:
                                if y_idx == x_idx:
                                    continue
                                n_free[y_idx] -= 1
                        elif update[1] == 'single':
                            y0 = update[2]
                            n_iso[y0] -= 1
                            
        return in_C, n_free, n_iso

    def try_local_search(in_C, n_free, n_iso, deadline, max_rounds=20):
        """
        Try local search polish: swap points to increase C.
        Returns (in_C, n_free, n_iso) after polish.
        """
        polish_improved = True
        polish_rounds = 0
        while polish_improved and polish_rounds < max_rounds:
            polish_improved = False
            polish_rounds += 1
            if time.time() > deadline - 5:
                break
                
            # Find a point in C that can be removed without violating Condition 1
            # and then try to add points
            for rem_idx in range(n_points):
                if not in_C[rem_idx]:
                    continue
                if time.time() > deadline - 5:
                    break
                    
                # Check if removing rem_idx is safe
                safe = True
                affected_lines = []
                
                for line_idx in point_to_lines[rem_idx]:
                    line = all_lines[line_idx]
                    c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
                    if len(c_pts) == 1 and c_pts[0] == rem_idx:
                        # After removal, this line becomes empty
                        for y_idx in line:
                            if y_idx == rem_idx:
                                continue
                            if n_free[y_idx] < 2:
                                safe = False
                                break
                        if not safe:
                            break
                        affected_lines.append(line_idx)
                        
                if not safe:
                    continue
                    
                # Try to add points not in C that can be added
                added = []
                for add_idx in range(n_points):
                    if in_C[add_idx]:
                        continue
                    if add_idx == rem_idx:
                        continue
                    if n_free[add_idx] < 1:
                        continue
                        
                    # Check if add_idx can be added (considering rem_idx is removed)
                    valid = True
                    for line_idx in point_to_lines[add_idx]:
                        line = all_lines[line_idx]
                        c_pts = [pt_idx for pt_idx in line if in_C[pt_idx] and pt_idx != rem_idx]
                        
                        if len(c_pts) == 0:
                            for y_idx in line:
                                if y_idx == add_idx or y_idx == rem_idx:
                                    continue
                                if n_free[y_idx] < 2:
                                    valid = False
                                    break
                            if not valid:
                                break
                        elif len(c_pts) == 1:
                            y0 = c_pts[0]
                            if n_iso[y0] < 2:
                                valid = False
                                break
                        else:
                            pass
                            
                    if valid:
                        added.append(add_idx)
                        if len(added) >= 2:
                            break
                            
                if len(added) >= 2:
                    # Perform the swap
                    in_C[rem_idx] = False
                    for add_idx in added:
                        in_C[add_idx] = True
                    polish_improved = True
                    break
                    
        return in_C, n_free, n_iso

    def compute_counts(in_C):
        """Compute n_free and n_iso from scratch."""
        n_free = [0] * n_points
        n_iso = [0] * n_points
        
        for line in all_lines:
            c_pts = [pt_idx for pt_idx in line if in_C[pt_idx]]
            if len(c_pts) == 0:
                for pt_idx in line:
                    n_free[pt_idx] += 1
            elif len(c_pts) == 1:
                n_iso[c_pts[0]] += 1
                
        return n_free, n_iso

    # Initial C: points on unital U with y NOT in Y
    # U: 2*xa + ya*ya - yb*yb*k == 0
    # Y: ya is a square mod p
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    # Best solution found
    best_C_size = -1
    best_in_C = None
    best_n_free = None
    best_n_iso = None
    
    # Try multiple random seeds for different initial orderings
    rng = random.Random(42)
    
    # First, try the deterministic initial C
    in_C = [False] * n_points
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            if (ya, yb) not in y_special:
                in_C[i] = True
                
    n_free, n_iso = compute_counts(in_C)
    
    # Run greedy
    in_C, n_free, n_iso = try_greedy(in_C, n_free, n_iso, deadline)
    
    # Run local search
    in_C, n_free, n_iso = try_local_search(in_C, n_free, n_iso, deadline)
    
    c_size = sum(in_C)
    if c_size > best_C_size:
        best_C_size = c_size
        best_in_C = in_C[:]
        best_n_free = n_free[:]
        best_n_iso = n_iso[:]
    
    # Try multiple random restarts with different initial C sets
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        # Create a random initial C set
        in_C = [False] * n_points
        
        # Start with a random subset of the unital points
        unital_points = []
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            if (2*xa + ya*ya - yb*yb*k) % p == 0:
                unital_points.append(i)
                
        # Randomly select some unital points
        rng.shuffle(unital_points)
        n_select = len(unital_points) // 2
        for idx in unital_points[:n_select]:
            in_C[idx] = True
            
        # Also randomly add some other points
        other_points = [i for i in range(n_points) if not in_C[i]]
        rng.shuffle(other_points)
        n_other = n_points // 10
        for idx in other_points[:n_other]:
            in_C[idx] = True
            
        n_free, n_iso = compute_counts(in_C)
        
        # Run greedy
        in_C, n_free, n_iso = try_greedy(in_C, n_free, n_iso, deadline)
        
        # Run local search
        in_C, n_free, n_iso = try_local_search(in_C, n_free, n_iso, deadline)
        
        c_size = sum(in_C)
        if c_size > best_C_size:
            best_C_size = c_size
            best_in_C = in_C[:]
            best_n_free = n_free[:]
            best_n_iso = n_iso[:]
            
    # Build result N = complement of C
    if best_in_C is None:
        # Fallback: empty C (N = all points)
        pts = all_points
    else:
        pts = [all_points[i] for i in range(n_points) if not best_in_C[i]]
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
