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
       We maintain coverage counts to efficiently check if a removal is valid.
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

    # Precompute line members for each point and direction.
    # line_members[i][d] is a numpy array of indices of points on the line through i in direction d, excluding i.
    # i = xa * 343 + xb * 49 + ya * 7 + yb
    line_members = np.zeros((n_points, n_dirs, q-1), dtype=np.int32)
    
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        pt = ((xa, xb), (ya, yb))
        for d in range(n_dirs):
            v = directions[d]
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
                line_members[i, d, t_idx-1] = j

    # Step 1: Construct initial S (excluded set) using the unital method.
    # U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a
    # Norm((a,b)) = a^2 - 3*b^2
    # Condition: 2*xa + ya^2 - 3*yb^2 = 0 (mod 7).
    
    # Special y set Y = {(u^2 mod p, v) : u, v in F_p}.
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    S = np.zeros(n_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = get_index(xa, xb, ya, yb)
                        S[idx] = True
                        
    # N is the complement of S.
    # We want to maximize |S|, so we want to add more points to S.
    # Start with the initial S, and greedily add more points.
    
    # valid_dirs[x] is the count of directions d such that L(x, d) \ {x} is fully in N.
    # is_valid[x][d] is True if L(x, d) \ {x} is fully in N.
    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    valid_dirs = np.zeros(n_points, dtype=np.int32)
    
    # Initialize is_valid and valid_dirs
    # is_valid[x][d] is False if any point in line_members[x][d] is in S.
    # We can compute this efficiently using broadcasting or loops.
    # Since n_points=2401, n_dirs=50, q=49.
    # For each x, for each d, check if np.any(S[line_members[x, d, :]])
    # This is 2401 * 50 checks. Each check involves 48 elements.
    # Total ~ 5.7M operations. Fast enough.
    
    for x in range(n_points):
        for d in range(n_dirs):
            # Check if any point in line_members[x, d] is in S
            if np.any(S[line_members[x, d, :]]):
                is_valid[x, d] = False
            else:
                valid_dirs[x] += 1
                
    # Check if initial set is valid
    # If any x has valid_dirs[x] == 0, the construction failed.
    # But the Hermitian unital construction is known to be valid.
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Greedy addition to S
    # We iterate through points. If adding p to S is valid, we add it.
    # To improve the score, we might want to try multiple passes or different orders.
    # However, a single pass is usually sufficient for this size.
    # Let's try a few passes with random shuffling to escape local minima?
    # Given the time limit, let's stick to one pass first, then see if we have time for more.
    
    # Actually, the order matters. Points that are "harder" to remove (i.e., break more lines)
    # should be considered later? Or earlier?
    # Usually, removing points that are "central" or have high degree in the constraint graph
    # is harder.
    # Let's just iterate in index order for determinism, or maybe shuffle?
    # The current program uses index order. Let's stick to that or try a few shuffles if time permits.
    
    # Optimization:
    # When checking if adding p is valid:
    # We need to check if any x has valid_dirs[x] drop to 0.
    # This happens if x has valid_dirs[x] == 1 and the only valid direction d is broken by p.
    # p breaks direction d for x if p is in L(x, d) \ {x}.
    # This is equivalent to x being in L(p, d) \ {p}.
    # So for a candidate p, we look at all x in L(p, d) \ {p} for all d.
    # For each such x, if is_valid[x, d] is True, then adding p would set is_valid[x, d] to False.
    # If valid_dirs[x] was 1, it becomes 0, which is invalid.
    
    # So the check is:
    # For each d:
    #   For each x in line_members[p, d]:
    #     If valid_dirs[x] == 1 and is_valid[x, d]:
    #       Return False (invalid)
    
    # If valid, update:
    # For each d:
    #   For each x in line_members[p, d]:
    #     If is_valid[x, d]:
    #       is_valid[x, d] = False
    #       valid_dirs[x] -= 1
    # S[p] = True
    
    # We can vectorize the check?
    # For a given p, the set of affected (x, d) pairs is the union over d of (line_members[p, d], d).
    # We can create a mask of affected x's for each d.
    
    for p in range(n_points):
        if time.time() > deadline:
            break
            
        if S[p]:
            continue
            
        # Check validity
        valid = True
        for d in range(n_dirs):
            members = line_members[p, d, :]
            # We need to check if any x in members has valid_dirs[x] == 1 and is_valid[x, d]
            # Let's get the indices of x in members where is_valid[x, d] is True
            # mask = is_valid[members, d]
            # if np.any(valid_dirs[members[mask]] == 1):
            #    valid = False
            #    break
            
            # More efficient:
            # Get x's in members that are currently valid for direction d
            # x_valid = members[is_valid[members, d]]
            # if np.any(valid_dirs[x_valid] == 1):
            #    valid = False
            #    break
            
            # Let's try to avoid creating temporary arrays if possible?
            # But numpy slicing is fast.
            
            x_valid = members[is_valid[members, d]]
            if len(x_valid) > 0:
                if np.any(valid_dirs[x_valid] == 1):
                    valid = False
                    break
                    
        if valid:
            S[p] = True
            # Update is_valid and valid_dirs
            for d in range(n_dirs):
                members = line_members[p, d, :]
                x_valid = members[is_valid[members, d]]
                if len(x_valid) > 0:
                    is_valid[x_valid, d] = False
                    valid_dirs[x_valid] -= 1

    # Construct N
    N_indices = np.where(~S)[0]
    
    # Create result array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
