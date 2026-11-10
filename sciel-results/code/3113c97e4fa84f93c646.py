# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Map index to (a, b)
    # index = a + b * 7
    
    def idx_to_ab(idx):
        return (idx % p, idx // p)

    def ab_to_idx(a, b):
        return a + b * p

    # Precompute lines
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # v is a pair (v_x, v_y) in F_49 x F_49
    # Represent v as (vx_a, vx_b, vy_a, vy_b)
    
    directions = []
    # v = (1, m)
    for m_idx in range(q):
        ma, mb = idx_to_ab(m_idx)
        # v_x = 1 = (1, 0)
        # v_y = m = (ma, mb)
        directions.append(((1, 0), (ma, mb)))
    # v = (0, 1)
    directions.append(((0, 0), (1, 0)))
    
    # Precompute lines: list of lists of point indices
    # lines[dir_idx][offset] = list of 49 point indices
    # Point index: x_idx * q + y_idx
    
    all_lines = []
    for v in directions:
        vx_a, vx_b = v[0]
        vy_a, vy_b = v[1]
        
        # We need to generate lines.
        # A line is determined by a starting point p0 and direction v.
        # To cover all lines, we can iterate over p0 in a fundamental domain.
        # Simpler: iterate over all p0, but deduplicate?
        # Or just iterate over all p0, and for each, generate the line.
        # There are q lines per direction.
        # We can pick representative p0s.
        # For v=(1, m), lines are disjoint.
        # We can pick p0 with x=0?
        # Let's just generate all q lines for this direction by picking p0 from a set of q representatives.
        # Representatives: points (0, y) for y in F_49?
        # If v_x != 0, x changes, so (0,y) are good reps.
        # If v_x == 0 (vertical), x is constant. Reps: (x, 0) for x in F_49.
        
        reps = []
        if vx_a != 0 or vx_b != 0:
            # v_x != 0
            for y_idx in range(q):
                reps.append((0, y_idx)) # x=0, y=y_idx
        else:
            # v_x == 0
            for x_idx in range(q):
                reps.append((x_idx, 0)) # x=x_idx, y=0
                
        dir_lines = []
        for p0 in reps:
            p0_x_idx, p0_y_idx = p0
            line_pts = []
            # p = p0 + t * v
            # t in F_49
            for t_idx in range(q):
                ta, tb = idx_to_ab(t_idx)
                # p_x = p0_x + t * v_x
                # p_y = p0_y + t * v_y
                # Addition in F_49: (a,b) + (c,d) = (a+c, b+d) mod 7
                # Multiplication in F_49: (a,b)*(c,d) = (ac + 3bd, ad + bc)
                
                # t * v_x
                tvx_a = (ta * vx_a + 3 * tb * vx_b) % p
                tvx_b = (ta * vx_b + tb * vx_a) % p
                # t * v_y
                tvy_a = (ta * vy_a + 3 * tb * vy_b) % p
                tvy_b = (ta * vy_b + tb * vy_a) % p
                
                # p_x = p0_x + tvx
                p0_xa, p0_xb = idx_to_ab(p0_x_idx)
                px_a = (p0_xa + tvx_a) % p
                px_b = (p0_xb + tvx_b) % p
                px_idx = ab_to_idx(px_a, px_b)
                
                # p_y = p0_y + tvy
                p0_ya, p0_yb = idx_to_ab(p0_y_idx)
                py_a = (p0_ya + tvy_a) % p
                py_b = (p0_yb + tvy_b) % p
                py_idx = ab_to_idx(py_a, py_b)
                
                pt_idx = px_idx * q + py_idx
                line_pts.append(pt_idx)
                
            dir_lines.append(line_pts)
            
        all_lines.append(dir_lines)
        
    # Search for best S
    # S is defined by X_set and Y_set
    # S = { (x,y) | x in X_set, y in Y_set }
    
    best_score = 0
    best_X = set()
    best_Y = set()
    
    # Try different sizes
    # X_set from first k elements of F_49
    # Y_set from first m elements of F_49
    
    def is_valid(X_set, Y_set):
        # Create boolean array for S
        S = np.zeros(q*q, dtype=bool)
        for x in X_set:
            for y in Y_set:
                S[x*q + y] = True
                
        # Compute counts for each line
        # line_counts[dir_idx][line_idx] = count of S in line
        line_counts = np.zeros((len(all_lines), len(all_lines[0])), dtype=int)
        
        for d_idx, dir_lines in enumerate(all_lines):
            for l_idx, line_pts in enumerate(dir_lines):
                cnt = 0
                for pt in line_pts:
                    if S[pt]:
                        cnt += 1
                line_counts[d_idx, l_idx] = cnt
                
        # Check each point
        for pt in range(q*q):
            x_idx = pt // q
            y_idx = pt % q
            in_S = S[pt]
            target_count = 1 if in_S else 0
            
            # Check all directions
            found_clean = False
            for d_idx in range(len(all_lines)):
                # Find which line in this direction contains pt
                # We can precompute line_index_for_point
                # But for simplicity, we can just check if pt is in the line?
                # Better: precompute mapping pt -> (d_idx, l_idx)
                # There are 50 lines through each pt.
                # We can precompute a list of (d_idx, l_idx) for each pt.
                
                pass # Placeholder, will implement mapping below
                
            # Implement mapping
            # line_map[pt] = list of (d_idx, l_idx)
            # We can build this once outside.
            
            return False # Placeholder
            
    # Precompute line_map
    line_map = [[] for _ in range(q*q)]
    for d_idx, dir_lines in enumerate(all_lines):
        for l_idx, line_pts in enumerate(dir_lines):
            for pt in line_pts:
                line_map[pt].append((d_idx, l_idx))
                
    def is_valid_fast(X_set, Y_set):
        S = np.zeros(q*q, dtype=bool)
        for x in X_set:
            for y in Y_set:
                S[x*q + y] = True
                
        line_counts = np.zeros((len(all_lines), len(all_lines[0])), dtype=int)
        for d_idx, dir_lines in enumerate(all_lines):
            for l_idx, line_pts in enumerate(dir_lines):
                cnt = 0
                for pt in line_pts:
                    if S[pt]:
                        cnt += 1
                line_counts[d_idx, l_idx] = cnt
                
        for pt in range(q*q):
            in_S = S[pt]
            target_count = 1 if in_S else 0
            
            for d_idx, l_idx in line_map[pt]:
                if line_counts[d_idx, l_idx] == target_count:
                    break
            else:
                return False
        return True

    # Search
    # Try increasing size of S
    # Start with known good: X size 7, Y size 21
    # Try X size 7, Y size 22, 23...
    # Try X size 8, Y size 20...
    
    candidates = []
    # Generate candidates
    for kx in range(1, 15):
        for ky in range(1, 30):
            if kx * ky > 2401:
                continue
            # Use first kx elements of X, first ky elements of Y
            X_set = set(range(kx))
            Y_set = set(range(ky))
            candidates.append((kx * ky, X_set, Y_set))
            
    # Sort by size descending
    candidates.sort(key=lambda x: -x[0])
    
    for size, X_set, Y_set in candidates:
        if size <= best_score:
            continue
        if is_valid_fast(X_set, Y_set):
            best_score = size
            best_X = X_set
            best_Y = Y_set
            
    # Construct final set
    N = []
    for x in range(q):
        for y in range(q):
            pt_idx = x*q + y
            if pt_idx not in [xx*q + yy for xx in best_X for yy in best_Y]:
                xa, xb = idx_to_ab(x)
                ya, yb = idx_to_ab(y)
                N.append([[xa, xb], [ya, yb]])
                
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
