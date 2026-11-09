import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    n_points = q * q  # 2401
    
    # Precompute field elements? No, just use (a, b) tuples.
    # But for speed, we can represent points as flat indices 0..2400.
    # Point i: xa = i // 343, xb = (i // 49) % 7, ya = (i // 7) % 7, yb = i % 7
    
    def get_coords(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb
    
    # Directions: (1, m) for m in F_q, and (0, 1)
    # m is represented as (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 0), (0, 1)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # Precompute line members for each point and direction?
    # This is 2401 * 50 lists of 48 ints.
    # Memory: 2401 * 50 * 48 * 4 bytes ~ 23 MB. Feasible.
    # Time to precompute: 2401 * 50 * 48 ~ 5.7M operations.
    # Let's precompute `line_members[i][d]` as a list of indices.
    
    line_members = [[None for _ in range(n_dirs)] for _ in range(n_points)]
    
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        for d in range(n_dirs):
            vx_a, vx_b = directions[d][0]
            vy_a, vy_b = directions[d][1]
            
            members = []
            for t in range(1, q):
                # x' = x + t*v
                # x_a' = (x_a + t*v_x_a) % 7
                # x_b' = (x_b + t*v_x_b) % 7
                # Note: t is in F_7? No, t is in F_49.
                # Wait, the line is {x + t*v : t in F_49}.
                # t is an element of F_49, so t = ta + tb*alpha.
                # t*v involves field multiplication.
                
                # The previous code used `t` as an integer 1..48 and did `(t * v[0][0]) % p`.
                # This is INCORRECT for field arithmetic.
                # t is in F_49. v is in F_49.
                # t * v is field multiplication.
                
                # Let's fix the field arithmetic for line generation.
                # t ranges over all non-zero elements of F_49.
                # We can iterate t_idx from 1 to 48.
                # t = t_a + t_b * alpha.
                # t_idx to (ta, tb): ta = t_idx // 7, tb = t_idx % 7.
                
                pass # Placeholder
            
            # This part needs correct field arithmetic.
            
    # Let's redefine the line generation correctly.
    
    def f_mul(a, b):
        # a, b are (a0, a1)
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    # Precompute all field elements as tuples
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    # Re-precompute line_members with correct field arithmetic
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        pt = ((xa, xb), (ya, yb))
        for d in range(n_dirs):
            v = directions[d]
            members = []
            for t_idx in range(1, q):
                t = field_elements[t_idx]
                # t*v
                tv_x = f_mul(t, v[0])
                tv_y = f_mul(t, v[1])
                # pt + tv
                new_x = f_add(pt[0], tv_x)
                new_y = f_add(pt[1], tv_y)
                # Convert new_x, new_y to index
                # new_x = (nxa, nxb), new_y = (nya, nyb)
                nxa, nxb = new_x
                nya, nyb = new_y
                j = nxa * 343 + nxb * 49 + nya * 7 + nyb
                members.append(j)
            line_members[i][d] = members
            
    # Initial state: S is empty, N is all points.
    # S is the set of excluded points.
    S = set()
    
    # cover[i][d] is True if no point in S is on the line through i in direction d (excluding i).
    # Initially, S is empty, so cover[i][d] is True for all i, d.
    cover = np.ones((n_points, n_dirs), dtype=bool)
    coverage = np.full(n_points, n_dirs, dtype=np.int32)
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Greedy addition
    # Order of addition matters. Random or sorted?
    # Let's try adding points in a fixed order, but maybe shuffle to avoid bias?
    # Or just iterate 0 to 2400.
    
    # To speed up checks, we can maintain a list of points that are "critical" (coverage == 1).
    # But updating that list is complex.
    # Simple check: for each candidate p, check if it breaks any x.
    
    # Optimization: Only check x that are on lines through p.
    # For a given p, the affected x are those such that p is on the line through x in some direction d.
    # This is equivalent to: x is on the line through p in direction d.
    # So we iterate d, get members of line(p, d), which are the x's.
    
    for p in range(n_points):
        if time.time() > deadline:
            break
            
        if p in S:
            continue
            
        # Check if adding p to S is valid
        valid = True
        # We need to check all x that would lose coverage.
        # x loses coverage in direction d if:
        # 1. x is on the line through p in direction d (i.e., p is in line_members[x][d])
        #    Which is equivalent to x is in line_members[p][d]?
        #    Line is symmetric. If p = x + t*v, then x = p - t*v = p + (-t)*v.
        #    So yes, x is in line_members[p][d].
        # 2. cover[x][d] is currently True.
        
        for d in range(n_dirs):
            members = line_members[p][d]
            for x in members:
                if cover[x][d]:
                    # x would lose this direction
                    if coverage[x] == 1:
                        valid = False
                        break
            if not valid:
                break
                
        if valid:
            S.add(p)
            # Update cover and coverage
            for d in range(n_dirs):
                members = line_members[p][d]
                for x in members:
                    if cover[x][d]:
                        cover[x][d] = False
                        coverage[x] -= 1
                        
    # Construct N
    N_indices = [i for i in range(n_points) if i not in S]
    
    # Create result array
    # Shape (k, 2, 2)
    # Each point is [[xa, xb], [ya, yb]]
    
    if len(N_indices) == 0:
        # Should not happen, but just in case
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
        
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
