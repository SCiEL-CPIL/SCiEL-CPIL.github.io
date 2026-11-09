# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Initialize N using the Hermitian unital construction.
    3. Precompute the 'affects' mapping: for each point r, which (p, d) pairs 
       are such that r lies on the line through p in direction d (r != p).
    4. Use a greedy multi-pass removal algorithm to minimize |N| while maintaining
       the Nikodym property.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    # Find smallest quadratic non-residue mod p
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3  # fallback

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    # f_add[x, y] = x + y
    # f_mul[x, y] = x * y
    # f_neg[x] = -x
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            # Addition
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            # Multiplication: (xa + xb*alpha)(ya + yb*alpha)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical
    n_dir = len(directions)

    # All points in F_q^2. Index: x_idx * q + y_idx
    # x_idx, y_idx are in 0..q-1 (these are the integer representations of field elements)
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))
    
    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Initialize N
    in_N = np.ones(q * q, dtype=bool)

    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Blocking set B: y such that y_a is a square mod p?
    # Current logic: y_special contains (u^2, v). So y_a is a square.
    # Excluded if in Unital AND y NOT in y_special.
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    in_N[point_index[(x_idx, y_idx)]] = False

    # not_in_N_count[p_idx, d_idx] = number of points on line(p, v_d) \ {p} that are NOT in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    # For each direction d, for each point p, count non-N points on line(p, d) \ {p}
    # Line(p, d) = { p + t*v_d : t in F_q }
    # Points other than p: t != 0
    
    for d_idx, (vx, vy) in enumerate(directions):
        # Precompute t*v for t in 1..q-1
        t_v = []
        for t in range(1, q):
            tx = f_mul[t, vx]
            ty = f_mul[t, vy]
            t_v.append((tx, ty))
        
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            count = 0
            for tx, ty in t_v:
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = point_index[(nx, ny)]
                if not in_N[n_idx]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    # cover_count[p_idx] = number of directions d such that not_in_N_count[p_idx, d] == 0
    cover_count = np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    # Precompute affects
    # affects[r_idx] = list of (p_idx, d_idx) such that r is on line(p, d) \ {p}
    # This means p + t*v_d = r for some t != 0
    # => p = r - t*v_d
    
    affects = [[] for _ in range(q * q)]
    for r_idx in range(q * q):
        rx, ry = idx_to_point[r_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                # p = r - t*v
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                neg_tv_x = f_neg[tv_x]
                neg_tv_y = f_neg[tv_y]
                px = f_add[rx, neg_tv_x]
                py = f_add[ry, neg_tv_y]
                if (px, py) in point_index:
                    p_idx = point_index[(px, py)]
                    affects[r_idx].append((p_idx, d_idx))

    # Greedy removal
    # We iterate through points in N and try to remove them.
    # We do multiple passes to ensure we don't miss removals due to ordering.
    
    while True:
        if time.time() > deadline - 10:
            break
            
        progress = False
        # Iterate through all indices. 
        # To be more efficient, we could sort by some heuristic, but simple iteration 
        # with multiple passes is robust.
        
        # We check if time is up inside the loop to avoid long stalls
        for r_idx in range(q * q):
            if not in_N[r_idx]:
                continue
            if time.time() > deadline - 5:
                break
                
            # Tentatively remove r_idx
            # Check if it causes any point p to lose all covering directions
            
            # We need to track which cover_counts change and by how much to revert if invalid
            # affected_cover_changes: list of (p_idx, delta_cover)
            # But we also need to revert not_in_N_count.
            
            # Let's collect the (p_idx, d_idx, old_count) for all affected pairs
            # and check validity on the fly.
            
            invalid = False
            affected = []
            
            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_count[p_idx, d_idx]
                new_count = old_count + 1
                
                # If old_count was 0, this direction no longer covers p_idx
                # So cover_count[p_idx] decreases by 1
                if old_count == 0:
                    if cover_count[p_idx] == 1:
                        # This point p_idx will have 0 covering directions
                        invalid = True
                        break
                    # else cover_count[p_idx] > 1, so it's still covered by at least one other direction
                    # We need to remember to decrement cover_count[p_idx] if we proceed
                
                affected.append((p_idx, d_idx, old_count))
            
            if invalid:
                continue # Revert is not needed as we didn't change anything
            
            # If not invalid, commit the changes
            for p_idx, d_idx, old_count in affected:
                not_in_N_count[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count[p_idx] -= 1
            
            in_N[r_idx] = False
            progress = True
        
        if not progress:
            break

    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return the base set (should not happen if initial set is valid)
        points = []
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if in_N[point_index[(x_idx, y_idx)]]:
                    points.append([[xa, xb], [ya, yb]])
                # Wait, if in_N is all false, we need to reconstruct base set
                # But in_N is modified in place. 
                # If we removed everything, it's invalid.
                # The fallback should reconstruct the *initial* base set.
                # But we lost the initial state. 
                # Let's assume we don't remove everything.
                # If points is empty, we must return a valid set.
                # The initial set was valid. 
                # So we should have kept a copy of the initial in_N or points.
                
    # To be safe, if points is empty, we return the initial construction again.
    # But we modified in_N. 
    # Let's store the initial points list before modification? 
    # That uses memory but is safe. 
    # Or just trust the algorithm doesn't remove all points. 
    # Given the structure, it shouldn't.
    
    # However, to be absolutely safe against edge cases where progress=True leads to empty set:
    if len(points) == 0:
        # Reconstruct initial base set
        in_N_init = np.ones(q * q, dtype=bool)
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0:
                    if y_idx not in y_special:
                        in_N_init[point_index[(x_idx, y_idx)]] = False
        
        points = []
        for i in range(q * q):
            if in_N_init[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
