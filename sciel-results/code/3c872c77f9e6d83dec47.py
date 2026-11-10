import numpy as np
from itertools import product
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    # Elements represented as (a, b) meaning a + b*alpha
    # Multiplication: (a,b)*(c,d) = (ac+3bd, ad+bc) mod 7
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*a[1]) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)

    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute all lines
    # Point: (xa, xb, ya, yb)
    all_lines = []
    seen_lines = set()
    
    # Vertical lines: x = (xa, xb), varying y
    for xa in range(p):
        for xb in range(p):
            points_on_line = []
            for ya in range(p):
                for yb in range(p):
                    points_on_line.append((xa, xb, ya, yb))
            line_tuple = tuple(sorted(points_on_line))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                all_lines.append(frozenset(points_on_line))
    
    # Non-vertical lines: direction (1, m) where m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            for x0a in range(p):
                for x0b in range(p):
                    for y0a in range(p):
                        for y0b in range(p):
                            points_on_line = []
                            for ta in range(p):
                                for tb in range(p):
                                    xa = (x0a + ta) % p
                                    xb = (x0b + tb) % p
                                    t_m = field_mul((ta, tb), (ma, mb))
                                    ya = (y0a + t_m[0]) % p
                                    yb = (y0b + t_m[1]) % p
                                    points_on_line.append((xa, xb, ya, yb))
                            
                            line_tuple = tuple(sorted(points_on_line))
                            if line_tuple not in seen_lines:
                                seen_lines.add(line_tuple)
                                all_lines.append(frozenset(points_on_line))
    
    # Build point-to-lines incidence
    point_to_lines = {}
    for point in product(range(p), repeat=4):
        point_to_lines[point] = []
    
    for i, line in enumerate(all_lines):
        for point in line:
            point_to_lines[point].append(i)
            
    # Precompute for each line, the list of points
    line_points = [list(line) for line in all_lines]
    
    # State variables
    # G[x] = number of good lines for x
    # line_state[i] = 0 (empty), 1 (singleton, store point in line_single[i]), 2 (multiple)
    # line_single[i] = the point if state is 1, else None
    
    all_points = list(product(range(p), repeat=4))
    point_index = {pt: i for i, pt in enumerate(all_points)}
    
    def try_add_point(p_idx, C, G, line_state, line_single):
        """
        Try to add point p_idx to C.
        Returns True if successful, False otherwise.
        Modifies C, G, line_state, line_single in place if successful.
        """
        p = all_points[p_idx]
        if p_idx in C:
            return False
            
        # Check if adding p is valid
        # We need to simulate the changes and see if any G[x] drops to 0
        
        # We'll collect the changes: list of (x_idx, delta)
        changes = []
        
        lines_through_p = point_to_lines[p]
        
        for l_idx in lines_through_p:
            state = line_state[l_idx]
            pts = line_points[l_idx]
            
            if state == 0:
                # Line was empty. Becomes singleton {p}.
                # For all x in pts, x != p:
                #   If x in N (which is all x not in C, and p is being added so p was in N):
                #   L was good for x (empty). Now L is not good (singleton p != x).
                #   So G[x] decreases by 1.
                # For p:
                #   L was good for p (empty). Now L is good for p (singleton p).
                #   G[p] stays same.
                
                for x in pts:
                    if x != p:
                        x_idx = point_index[x]
                        # x is in N (since only p is being added, and x != p, x is not in C)
                        # Check if G[x] - 1 >= 1, i.e., G[x] >= 2
                        if G[x_idx] < 2:
                            return False
                        changes.append((x_idx, -1))
                        
                # Update state
                line_state[l_idx] = 1
                line_single[l_idx] = p
                
            elif state == 1:
                # Line was singleton {y}. Becomes multiple {y, p}.
                y = line_single[l_idx]
                # For y:
                #   L was good for y (singleton y). Now L is not good (multiple).
                #   So G[y] decreases by 1.
                y_idx = point_index[y]
                if G[y_idx] < 2:
                    return False
                changes.append((y_idx, -1))
                
                # For other x in pts, x != y, x != p:
                #   L was not good for x (singleton y != x).
                #   L is not good for x (multiple).
                #   No change.
                
                # For p:
                #   L was not good for p (singleton y != p).
                #   L is not good for p (multiple).
                #   No change.
                
                # Update state
                line_state[l_idx] = 2
                line_single[l_idx] = None
                
            else: # state == 2
                # Line was multiple. Stays multiple.
                # No one was using this line as a good line.
                # No changes to G.
                pass
        
        # If we got here, all checks passed. Apply changes.
        C.add(p_idx)
        for x_idx, delta in changes:
            G[x_idx] += delta
            
        return True

    def run_greedy(seed):
        random.seed(seed)
        C = set()
        G = [50] * len(all_points) # All lines are good for all points initially
        line_state = [0] * len(all_lines)
        line_single = [None] * len(all_lines)
        
        candidates = list(range(len(all_points)))
        random.shuffle(candidates)
        
        # We can do multiple passes to pick up points that became addable
        # Or just one pass? One pass is O(N * 50 * 49) ~ 12M ops. Fast enough.
        # But maybe a point rejected early becomes addable later?
        # No, adding points to C only *removes* good lines from others.
        # It never *adds* good lines to others.
        # So if a point p is rejected, it will remain rejected forever?
        # Let's check.
        # G[x] only decreases.
        # The condition for adding p depends on G[x] for neighbors.
        # Since G[x] only decreases, if p was rejected because some neighbor x had G[x] < 2,
        # G[x] will only get smaller, so p will still be rejected.
        # So a single pass in a random order is sufficient?
        # Wait. The condition for p depends on the *current* state of lines.
        # If a line is "empty", it contributes to G[x].
        # If we add points to C, lines become non-empty.
        # So G[x] decreases.
        # So yes, if p is rejected, it stays rejected.
        # Thus, one pass is optimal for a given order.
        
        for p_idx in candidates:
            try_add_point(p_idx, C, G, line_state, line_single)
            
        return C

    # Try multiple random restarts and keep the best (largest C)
    best_C = set()
    best_size = 0
    
    # Number of restarts. Each restart is ~12M ops.
    # 600s limit. Python is slow. Let's try 10-20 restarts.
    # Actually, let's profile mentally.
    # 2401 points. 50 lines each. 48 points per line.
    # Inner loop ~ 2400 checks.
    # 2401 * 2400 ~ 5.7M operations.
    # In Python, 5M ops might take 1-2 seconds.
    # So 10 restarts should take ~20s. Safe.
    
    for seed in range(20):
        C = run_greedy(seed)
        if len(C) > best_size:
            best_size = len(C)
            best_C = C
            
    # Construct N = all_points - C
    N_points = set(all_points)
    for p_idx in best_C:
        N_points.remove(all_points[p_idx])
        
    points_list = []
    for pt in sorted(N_points):
        points_list.append([[pt[0], pt[1]], [pt[2], pt[3]]])
        
    if not points_list:
        for pt in sorted(all_points):
            points_list.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(points_list, dtype=np.int64)
