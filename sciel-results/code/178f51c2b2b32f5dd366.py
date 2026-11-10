# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49, p = 7.
    
    Strategy: Start with S = full Hermitian unital (large complement), then greedily
    remove points from S that violate the constraint: for every x in S and every 
    direction v, there must exist t != 0 with x + t*v also in S.
    
    The resulting N = F_q^2 \\ S is a Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3, non-residue mod 7
    
    # Precompute field operations
    # Element (a, b) = a + b*alpha, alpha^2 = 3
    # Addition: (a1,b1) + (a2,b2) = (a1+a2, b1+b2) mod p
    # Multiplication: (a1,b1)*(a2,b2) = (a1*a2 + b1*b2*k, a1*b2 + a2*b1) mod p
    
    # Directions: v = (1, m) for m in F_q (49 directions), and v = (0, 1) (1 direction)
    # Total: 50 directions
    # Represent direction as (dx_a, dx_b, dy_a, dy_b)
    
    # Build all 2401 points as indices
    # Point index: ((xa * p + xb) * q) + (ya * p + yb) = xa * p * q + xb * q + ya * p + yb
    # Or more simply: flat index from 0 to 2400
    
    # Create arrays for all points
    xa_arr = np.arange(p)
    xb_arr = np.arange(p)
    ya_arr = np.arange(p)
    yb_arr = np.arange(p)
    
    XA, XB, YA, YB = np.meshgrid(xa_arr, xb_arr, ya_arr, yb_arr, indexing='ij')
    
    # Flatten to get all 2401 points
    all_xa = XA.ravel()
    all_xb = XB.ravel()
    all_ya = YA.ravel()
    all_yb = YB.ravel()
    
    # Map from (xa, xb, ya, yb) to flat index
    # flat_index = xa * q * p + xb * q + ya * p + yb
    # Wait, let me be more careful.
    # xa in 0..6, xb in 0..6, ya in 0..6, yb in 0..6
    # flat_index = xa * 49 * 7 + xb * 49 + ya * 7 + yb = xa * 343 + xb * 49 + ya * 7 + yb
    
    # Verify: total = 7 * 343 + 6 * 49 + 6 * 7 + 6 = 2401 + 294 + 42 + 6 = 2743? No.
    # Actually: max index = 6*343 + 6*49 + 6*7 + 6 = 2058 + 294 + 42 + 6 = 2400. Good.
    
    # Create lookup: for each flat index, store (xa, xb, ya, yb)
    point_coords = np.column_stack([all_xa, all_xb, all_ya, all_yb])
    
    # Also create reverse lookup: (xa, xb, ya, yb) -> flat index
    # flat_index = xa * 343 + xb * 49 + ya * 7 + yb
    def point_to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    # Precompute for each direction and each point, the other points on the line
    # Direction v = (dx_a, dx_b, dy_a, dy_b)
    # Line through point (xa, xb, ya, yb) in direction v:
    # (xa + t*dx_a, xb + t*dx_b, ya + t*dy_a, yb + t*dy_b) for t in F_q
    
    # For efficiency, precompute for each direction and each point, the set of other points on the line
    
    # Directions:
    # Type 1: v = (1, m) where m = (ma, mb) in F_q
    #   dx_a = 1, dx_b = 0, dy_a = ma, dy_b = mb
    # Type 2: v = (0, 1)
    #   dx_a = 0, dx_b = 0, dy_a = 0, dy_b = 1
    
    directions = []
    # Type 1: v = (1, m)
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    # Type 2: v = (0, 1)
    directions.append((0, 0, 0, 1))
    
    num_dirs = len(directions)  # 50
    
    # For each direction and each point, precompute the q-1 other points on the line
    # line_points[dir_idx][point_idx] = array of q-1 point indices (excluding the point itself)
    
    # This is 50 * 2401 * 48 entries, which is manageable
    
    # Actually, let's be smarter. For a given direction v and point x, the line is:
    # x + t*v for t in F_q, t != 0
    # So we need to compute x + t*v for t = 1, 2, ..., q-1 (in field arithmetic)
    
    # Let's precompute field multiplication for efficiency
    # mul[a, b] = a * b in F_q, where a, b are flat indices 0..48
    # Element a = (a // p) + (a % p) * alpha? No, let me use (a_a, a_b) = (a // p, a % p)
    # Wait, I defined element as (a, b) = a + b*alpha, with a, b in 0..6
    # Let me use index = a * p + b = a * 7 + b
    
    # So element index e: a = e // p, b = e % p
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for e1 in range(q):
        a1 = e1 // p
        b1 = e1 % p
        for e2 in range(q):
            a2 = e2 // p
            b2 = e2 % p
            # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
            # = (a1*a2 + b1*b2*k) + (a1*b2 + a2*b1)*alpha
            result_a = (a1 * a2 + b1 * b2 * k_const) % p
            result_b = (a1 * b2 + a2 * b1) % p
            mul_table[e1, e2] = result_a * p + result_b
    
    # For each direction and point, precompute the line points (excluding the point itself)
    # line_pts[dir_idx][pt_idx] = np.array of q-1 flat indices
    
    # This will take some time but is worth it for the greedy repair
    line_pts = [None] * num_dirs
    
    for d_idx, (dxa, dxb, dya, dyb) in enumerate(directions):
        # dxa, dxb are components of dx (the x-component of direction)
        # dya, dyb are components of dy (the y-component of direction)
        # Direction vector: (dx, dy) where dx = dxa + dxb*alpha, dy = dya + dyb*alpha
        
        # For point (xa, xb, ya, yb), the line points are:
        # (xa + t*dx_a, xb + t*dx_b, ya + t*dy_a, yb + t*dy_b)
        # where t is in F_q, t != 0
        
        # t = (ta, tb) = (ta + tb*alpha)
        # t*dx = (ta + tb*alpha) * (dxa + dxb*alpha) 
        # = (ta*dxa + tb*dxb*k) + (ta*dxb + tb*dxa)*alpha
        # So t*dx_a = (ta*dxa + tb*dxb*k) % p
        #    t*dx_b = (ta*dxb + tb*dxa) % p
        
        # Similarly for t*dy
        
        # Precompute for each t (as flat index 0..48, t != 0), the offsets
        t_offsets_xa = np.zeros(q, dtype=np.int32)
        t_offsets_xb = np.zeros(q, dtype=np.int32)
        t_offsets_ya = np.zeros(q, dtype=np.int32)
        t_offsets_yb = np.zeros(q, dtype=np.int32)
        
        for t_idx in range(q):
            ta = t_idx // p
            tb = t_idx % p
            # t * dx
            t_offsets_xa[t_idx] = (ta * dxa + tb * dxb * k_const) % p
            t_offsets_xb[t_idx] = (ta * dxb + tb * dxa) % p
            # t * dy
            t_offsets_ya[t_idx] = (ta * dya + tb * dyb * k_const) % p
            t_offsets_yb[t_idx] = (ta * dyb + tb * dya) % p
        
        # For each point, compute the line points
        line_pts[d_idx] = [None] * (q * q)
        
        for pt_idx in range(q * q):
            xa = point_coords[pt_idx, 0]
            xb = point_coords[pt_idx, 1]
            ya = point_coords[pt_idx, 2]
            yb = point_coords[pt_idx, 3]
            
            other_pts = []
            for t_idx in range(1, q):  # t != 0
                new_xa = (xa + t_offsets_xa[t_idx]) % p
                new_xb = (xb + t_offsets_xb[t_idx]) % p
                new_ya = (ya + t_offsets_ya[t_idx]) % p
                new_yb = (yb + t_offsets_yb[t_idx]) % p
                other_pt_idx = new_xa * 343 + new_xb * 49 + new_ya * 7 + new_yb
                other_pts.append(other_pt_idx)
            
            line_pts[d_idx][pt_idx] = np.array(other_pts, dtype=np.int32)
    
    # Initialize S with the full Hermitian unital
    # Unital condition: 2*x_a + y_a^2 - y_b^2*k = 0 (mod p)
    in_unital = (2 * all_xa + all_ya**2 - all_yb**2 * k_const) % p == 0
    
    S = in_unital.copy()  # Boolean array of size 2401
    s_count = S.sum()
    
    # Greedy repair: iteratively remove points from S that violate the constraint
    # A point x in S violates if there exists a direction v such that for all t != 0,
    # x + t*v is NOT in S
    
    # We'll iterate until no more violations or time runs out
    
    changed = True
    iteration = 0
    while changed and (time.time() < deadline):
        changed = False
        iteration += 1
        
        # Find all violating points
        # For each point in S, check if it violates
        violating = np.zeros(q * q, dtype=bool)
        
        for pt_idx in range(q * q):
            if not S[pt_idx]:
                continue
            
            # Check all directions
            for d_idx in range(num_dirs):
                other_pts = line_pts[d_idx][pt_idx]
                # Check if any of the other points is in S
                if not S[other_pts].any():
                    violating[pt_idx] = True
                    break
        
        if not violating.any():
            break
        
        # Remove violating points
        S[violating] = False
        changed = True
    
    # Build the result: N = complement of S
    N_mask = ~S
    n_count = N_mask.sum()
    
    if n_count == 0:
        # Should not happen, but just in case
        # Return a minimal valid set
        result = np.zeros((1, 2, 2), dtype=np.int64)
        return result
    
    # Get the points in N
    n_points = point_coords[N_mask]
    result = n_points.reshape(-1, 2, 2).astype(np.int64)
    
    return result


# EVOLVE-BLOCK-END
