import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    k = 3

    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin

    # Field operations
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def field_to_int(f):
        return f[0] * p + f[1]
        
    def int_to_field(i):
        return (i // p, i % p)

    # Encode/Decode
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def decode(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        xa = idx // p
        return (xa, xb, ya, yb)

    # Precompute line_id
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    coords = np.zeros((n_points, 4), dtype=np.int32)
    for idx in range(n_points):
        xa, xb, ya, yb = decode(idx)
        coords[idx] = [xa, xb, ya, yb]
        
    for d in range(n_dirs):
        if d < n_dirs - 1:
            ma = d // p
            mb = d % p
            m = (ma, mb)
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                x = (xa, xb)
                y = (ya, yb)
                mx = field_mul(m, x)
                c = field_add(y, field_neg(mx))
                line_id[idx, d] = field_to_int(c)
        else:
            for idx in range(n_points):
                xa, xb, ya, yb = coords[idx]
                line_id[idx, d] = field_to_int((xa, xb))

    # Precompute line members
    line_members = [None] * n_dirs
    for d in range(n_dirs):
        line_members[d] = [None] * q
        for lid in range(q):
            mask = line_id[:, d] == lid
            line_members[d][lid] = np.where(mask)[0]

    def is_valid(S):
        # S is the set N (the Nikodym set). We want to check if every point x has a line L through x such that L\{x} subset S.
        # Equivalently: for every x, exists d such that count(S on line(d, x)) == 1 if x in S, or count(S on line(d, x)) == 0 if x not in S?
        # Wait, the condition is: "every point x ... must lie on a line ... whose points other than x all belong to N."
        # Let N be the set returned.
        # If x in N: We need a line L through x such that L \ {x} subset N. This means all other points on L are in N.
        #   Let L = {x} union R. R subset N.
        #   This is satisfied if the number of points in L \ N is 1 (which is x) or 0 (if x is the only one not in N? No, x IS in N).
        #   Actually, if x in N, and L \ {x} subset N, then ALL points in L are in N. So count(N on L) == |L| = 49.
        # If x not in N: We need a line L through x such that L \ {x} subset N.
        #   x not in N. L \ {x} subset N. So the only point in L not in N is x.
        #   So count(N on L) == |L| - 1 = 48.
        
        # So the condition is: For every x, exists d such that:
        # if x in N: count(N on line(d,x)) == 49
        # if x not in N: count(N on line(d,x)) == 48
        
        satisfied = np.zeros(n_points, dtype=bool)
        for d in range(n_dirs):
            lids = line_id[:, d]
            counts = np.bincount(lids[S], minlength=q)
            counts_for_points = counts[lids]
            
            # If S[y] is True, we need count == 49
            # If S[y] is False, we need count == 48
            valid_for_d = (S & (counts_for_points == 49)) | (~S & (counts_for_points == 48))
            satisfied |= valid_for_d
            
            if np.all(satisfied):
                return True
        return False

    # We want to find a small N (minimize |N|).
    # Start with N = All points.
    # Try to remove points from N.
    # If removing x from N breaks validity for some point y, we might keep x.
    # But removing x might help other points? No, removing points from N makes it HARDER to satisfy "L\{x} subset N".
    # Because "L\{x} subset N" requires MORE points in N.
    # So if N is valid, any superset of N is valid?
    # If x in N and L\{x} subset N, then for any N' superset N, L\{x} subset N'. So x is still valid.
    # If x not in N and L\{x} subset N, then L\{x} subset N'. So x is still valid.
    # Yes, validity is monotone with respect to inclusion. If N is valid, any N' superset N is valid.
    # So we want to find a MINIMAL valid set N.
    # We can start with N = All, and try to REMOVE points.
    # If removing x keeps N valid, we remove it.
    # This is a Greedy Set Cover / Minimum Set Basis approach? No, it's just greedy deletion.
    
    N = np.ones(n_points, dtype=bool)
    # Shuffle points to try removals in random order
    all_points = list(range(n_points))
    random.shuffle(all_points)
    
    best_N = N.copy()
    best_size = n_points
    
    # Greedy Deletion
    for x in all_points:
        if time.time() > deadline:
            break
        if not N[x]:
            continue
        # Try removing x
        N[x] = False
        if is_valid(N):
            # Valid! Keep it removed
            current_size = np.sum(N)
            if current_size < best_size:
                best_size = current_size
                best_N = N.copy()
        else:
            # Invalid, put it back
            N[x] = True
    
    # Try a few more rounds of removal?
    # Since it's greedy, order matters.
    # But time is limited.
    
    # Use best_N found
    N = best_N
    
    # If N is empty (shouldn't happen), fallback
    if not np.any(N):
        N = np.ones(n_points, dtype=bool)
        
    # Convert to array
    N_points = []
    for idx in range(n_points):
        if N[idx]:
            xa, xb, ya, yb = decode(idx)
            N_points.append([[xa, xb], [ya, yb]])
            
    if not N_points:
        N_points = [[[0,0],[0,0]]]
        
    return np.array(N_points, dtype=np.int64)
