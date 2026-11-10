import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # Field operations
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def field_mul(x, y):
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)
    
    def field_neg(x):
        return ((-x[0])%p, (-x[1])%p)
    
    def field_sub(x, y):
        return field_add(x, field_neg(y))
    
    # Enumerate all field elements
    field_elements = [(a,b) for a in range(p) for b in range(p)]
    
    # All points in F_49^2
    all_points = [((xa,xb),(ya,yb)) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    n_points = 2401
    
    # Build lines
    lines = []  # list of lists of point indices
    
    # Vertical lines: direction (0,1)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    line_pts.append(point_to_idx[pt])
            lines.append(line_pts)
    
    # Slanted lines: direction (1,m), y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            line_pts.append(point_to_idx[pt])
                    lines.append(line_pts)
    
    n_lines = len(lines)  # 2450
    
    # For each point, which lines pass through it
    point_lines = [[] for _ in range(n_points)]
    for li, line in enumerate(lines):
        for pt_idx in line:
            point_lines[pt_idx].append(li)
    
    # Initial complement: Hermitian unital points with ya not in squares
    squares_mod7 = {0, 1, 2, 4}
    C = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2*xa + ya*ya - k*yb*yb) % p == 0
                    if in_unital and ya not in squares_mod7:
                        C.add(point_to_idx[((xa,xb),(ya,yb))])
    
    # Verify initial C
    def verify_nikodym(C_set):
        C_frozen = frozenset(C_set)
        for pt_idx in range(n_points):
            found = False
            for li in point_lines[pt_idx]:
                line = lines[li]
                valid = True
                for other in line:
                    if other != pt_idx and other in C_frozen:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                return False
        return True
    
    assert verify_nikodym(C), "Initial C should be valid"
    
    # Greedy expansion using good-line counting
    # c_L = number of C points on line L
    c_L = np.zeros(n_lines, dtype=np.int32)
    for pt in C:
        for li in point_lines[pt]:
            c_L[li] += 1
    
    # good_lines[x] = number of good lines for point x
    good_lines = np.zeros(n_points, dtype=np.int32)
    for x in range(n_points):
        for li in point_lines[x]:
            if x in C:
                if c_L[li] == 1:
                    good_lines[x] += 1
            else:
                if c_L[li] == 0:
                    good_lines[x] += 1
    
    # All points should have at least 1 good line
    assert np.all(good_lines >= 1), "Initial C invalid"
    
    best_C = set(C)
    best_size = len(C)
    
    # Candidates: all points not in C
    candidates = [pt for pt in range(n_points) if pt not in C]
    
    # Try to add candidates greedily
    # For efficiency, shuffle and try in random order, or try all
    # We'll try all candidates, and for each, check if it can be added
    
    # To make it faster, we can try in batches and re-verify periodically
    # But let's first try a simple greedy: sort candidates by some heuristic
    
    # Heuristic: prefer points that are on fewer "tight" lines
    # A point p can be added iff for every line L through p:
    #   - If c_L[li] == 0: all other points on L (which are not in C) must have good_lines > 1
    #     (because they lose L as a good line)
    #   - If c_L[li] == 1: the C point on L must have good_lines > 1
    #   - If c_L[li] >= 2: no effect on good_lines of other points (they already didn't count L)
    #   - For p itself: no change in good_lines (as shown above)
    
    # Let's implement this check
    def can_add(p, c_L, good_lines, C_set):
        """Check if point p can be added to C without making any point have 0 good lines."""
        for li in point_lines[p]:
            old_c = c_L[li]
            if old_c == 0:
                # All other points on this line (not in C, since c=0) lose this good line
                for other in lines[li]:
                    if other != p:
                        if good_lines[other] <= 1:
                            return False
            elif old_c == 1:
                # The single C point on this line loses this good line
                for other in lines[li]:
                    if other != p and other in C_set:
                        if good_lines[other] <= 1:
                            return False
            # old_c >= 2: no effect
        return True
    
    # Greedy: try to add candidates
    # Use a priority: try points that "damage" the least
    # For now, just iterate through all candidates
    
    # To speed up, we can do multiple passes
    # In each pass, try to add as many points as possible
    
    max_passes = 5
    for pass_num in range(max_passes):
        if time.time() - start_time > deadline - 30:
            break
        
        added_any = False
        # Try candidates in order
        for p in candidates:
            if time.time() - start_time > deadline - 30:
                break
            if p in C:
                continue
            if can_add(p, c_L, good_lines, C):
                # Add p to C
                C.add(p)
                added_any = True
                # Update c_L
                for li in point_lines[p]:
                    c_L[li] += 1
                # Update good_lines for affected points
                for li in point_lines[p]:
                    old_c = c_L[li] - 1  # c before adding p
                    if old_c == 0:
                        # Other non-C points on this line lose a good line
                        for other in lines[li]:
                            if other != p and other not in C:
                                good_lines[other] -= 1
                    elif old_c == 1:
                        # The C point on this line loses a good line
                        for other in lines[li]:
                            if other != p and other in C:
                                good_lines[other] -= 1
                # p itself: good_lines doesn't change (as proven)
        
        if not added_any:
            break
        
        best_C = set(C)
        best_size = len(C)
    
    # Final verification
    if not verify_nikodym(C):
        C = set(best_C)
    
    # Convert to array
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in C:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])
    
    return np.array(N_points, dtype=np.int64)
