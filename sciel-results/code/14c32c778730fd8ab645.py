import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Strategy:
    1. Start with a known valid Nikodym set based on the Hermitian unital construction.
       The excluded set S is points (x, y) in the unital U where y is NOT in the
       special set Y = {(u^2 mod p, v) : u, v in F_p}.
       N = F_q^2 \ S.
    2. Greedily try to remove points from N (add them to S) to increase the complement size.
       A point p can be removed if for every direction d, there exists at least one other
       point on the line through p in direction d that is already in S (or p itself is not
       required to cover any x that has no other covering direction).
       Actually, the condition is: for every point x in F_q^2, there must be a direction d
       such that the line L(x, d) \ {x} is fully contained in N.
       If we remove p from N (add to S), we break the line L(x, d) for any x such that p in L(x, d) \ {x}.
       This is equivalent to x in L(p, d) \ {p}.
       So removing p is valid if: for all x in F_q^2, if p in L(x, d) \ {x}, then there exists
       some direction d' (possibly d) such that L(x, d') \ {x} is still fully in N.
       
       More efficiently:
       Maintain `cover[x]` = number of directions d such that L(x, d) \ {x} is fully in N.
       Initially, cover[x] >= 1 for all x (since N is a valid Nikodym set).
       When we remove p, for each direction d, and each x in L(p, d) \ {p}:
           If L(x, d) \ {x} was fully in N before, and p was in it, then it is no longer fully in N.
           So cover[x] decreases by 1.
           If cover[x] becomes 0, the removal is invalid.
       
       We can precompute line members for all points and directions.
       Then for each candidate p, check if removal is valid by checking if any x would have cover[x] drop to 0.
       If valid, remove p and update cover counts.
    """
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, since 3 is a quadratic non-residue mod 7.
    
    n_points = q * q  # 2401
    
    # Directions: (1, m) for m in F_q, and (0, 1).
    # m is represented as (ma, mb).
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 0), (0, 1)))  # (0, 1)
    n_dirs = len(directions)  # 50
    
    # Precompute field operations
    def f_mul(a, b):
        # a, b are (a0, a1)
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    # Field elements indexed 0..48
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    def get_coords(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb
        
    def get_index(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    line_members = [[None for _ in range(n_dirs)] for _ in range(n_points)]
    
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
                nxa, nxb = new_x
                nya, nyb = new_y
                j = get_index(nxa, nxb, nya, nyb)
                members.append(j)
            line_members[i][d] = members
            
    # Step 1: Construct initial S (excluded set) using the unital method.
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = get_index(xa, xb, ya, yb)
                        S.add(idx)
                        
    # Initialize valid_dirs
    valid_dirs = np.zeros(n_points, dtype=np.int32)
    # is_valid[x][d] = True if L(x, d) \ {x} subset N_initial
    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    
    for x in range(n_points):
        for d in range(n_dirs):
            # Check if any point in line_members[x][d] is in S
            # If yes, is_valid[x][d] = False
            for y in line_members[x][d]:
                if y in S:
                    is_valid[x][d] = False
                    break
            if is_valid[x][d]:
                valid_dirs[x] += 1
                
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Greedy addition to S with multiple passes and different orderings
    # To improve the score, we want to add as many points as possible.
    # A point p can be added if for all x such that p in L(x, d) \ {x}, valid_dirs[x] > 1 OR the direction d is not the only valid one for x.
    # More precisely: adding p to S makes is_valid[x][d] = False for all x, d such that p in L(x, d) \ {x} and is_valid[x][d] was True.
    # This decreases valid_dirs[x] by 1 for each such (x, d).
    # The addition is valid if no x has valid_dirs[x] drop to 0.
    
    # Precompute for each p, the list of (x, d) pairs that would be affected.
    # p in L(x, d) \ {x}  <=>  x in L(p, d) \ {p}
    # So for each p, and each d, the affected x's are line_members[p][d].
    
    candidates = list(range(n_points))
    
    # Multiple passes with different orderings
    for pass_num in range(5):
        if time.time() > deadline:
            break
        
        if pass_num == 0:
            order = candidates
        elif pass_num == 1:
            order = list(reversed(candidates))
        else:
            # Random shuffle
            order = candidates[:]
            np.random.shuffle(order)
        
        for p in order:
            if time.time() > deadline:
                break
                
            if p in S:
                continue
                
            # Check if adding p is valid
            valid = True
            # We need to check if any x would have valid_dirs[x] become 0.
            # Only x with valid_dirs[x] == 1 are at risk.
            # For such x, if the only valid direction d has p in L(x, d) \ {x}, then x becomes invalid.
            
            for d in range(n_dirs):
                members = line_members[p][d]
                for x in members:
                    if valid_dirs[x] == 1 and is_valid[x][d]:
                        # x has only this direction d as valid, and p is on that line.
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                S.add(p)
                # Update is_valid and valid_dirs
                for d in range(n_dirs):
                    members = line_members[p][d]
                    for x in members:
                        if is_valid[x][d]:
                            is_valid[x][d] = False
                            valid_dirs[x] -= 1
                            
    # Construct N
    N_indices = [i for i in range(n_points) if i not in S]
    
    # Create result array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
