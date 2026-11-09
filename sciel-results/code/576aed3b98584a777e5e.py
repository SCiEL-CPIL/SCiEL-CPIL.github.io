import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Safety margin: 630s limit, we stop at 580s
    deadline = time.time() + 580.0

    # Field arithmetic in F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3
    
    # Precompute field operations for speed
    # idx_to_elem: index 0..48 -> (a, b)
    idx_to_elem = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {(a, b): i for i, (a, b) in enumerate(idx_to_elem)}

    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def f_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        aa = (a1 * a2 + b1 * b2 * k_const) % p
        bb = (a1 * b2 + b1 * a2) % p
        return (aa, bb)

    # Directions: v = (0, 1) and v = (1, m) for m in F_49
    # Total 50 directions
    directions = []
    directions.append(((0, 0), (1, 0))) # (dx, dy) -> (0,1)
    for m_idx in range(q):
        m_elem = idx_to_elem[m_idx]
        directions.append(((1, 0), m_elem)) # (1, m)
        
    num_dirs = len(directions) # 50

    # Precompute line points for each point x and direction v
    # line_pts[x][v] is an array of q-1 point indices (excluding x)
    # We also need to know, for a given point z, which (x, v) pairs are affected by z.
    # affected[z] is a list of (x, v) such that z is in line_pts[x][v]
    
    # Optimization: Instead of storing full line_pts for all x,v (2401*50*48 ints ~ 2.8M ints, ~11MB),
    # we can compute on the fly or store compactly. 
    # However, for pruning, we need to check if removing z breaks validity of x.
    # x is valid if there exists v such that all points in line_pts[x][v] are in N.
    # Let cnt_not_in_N[x][v] be the number of points in line_pts[x][v] NOT in N.
    # x is valid if min(cnt_not_in_N[x]) == 0.
    # Initially N = All, so cnt_not_in_N[x][v] = 0 for all x, v.
    # If we remove z from N, we increment cnt_not_in_N[x][v] for all (x, v) where z is in line_pts[x][v].
    # x becomes invalid if min(cnt_not_in_N[x]) > 0.
    # Wait, if we remove z, cnt_not_in_N increases.
    # Initially min is 0. After removal, if the unique valid direction for x contained z, its count becomes 1.
    # If x had another direction with count 0, it remains valid.
    # So x becomes invalid ONLY IF:
    # 1. x was valid (min cnt == 0)
    # 2. The set of directions v for which cnt_not_in_N[x][v] == 0 becomes empty after incrementing the specific v's containing z.
    # Actually, simpler:
    # x is valid if there is at least one v with cnt_not_in_N[x][v] == 0.
    # When we remove z, we increment cnt_not_in_N[x][v] for all (x, v) in affected[z].
    # x becomes invalid if, for ALL v in affected[z], the new cnt_not_in_N[x][v] > 0 AND for all v NOT in affected[z], cnt_not_in_N[x][v] > 0.
    # Since we start with all cnts 0, and only increment, any v not in affected[z] stays 0.
    # So x becomes invalid ONLY IF z is in EVERY line L(x, v) for all v such that L(x, v) \ {x} subset N.
    # Initially, all lines are "good" (cnt 0).
    # If we remove z, for a specific x, the lines L(x, v) that contain z will have cnt become 1.
    # The lines L(x, v) that do NOT contain z will have cnt remain 0.
    # So x remains valid as long as there is at least one direction v such that z is NOT in L(x, v).
    # When is z in L(x, v)? 
    # L(x, v) = { x + t*v | t in F_49 }.
    # z is in L(x, v) iff z - x is a multiple of v.
    # Since v is a vector (dx, dy), z-x is a multiple of v iff z-x = t*v for some t.
    # If v = (0, 1), z-x = (0, t). So x_a = z_a.
    # If v = (1, m), z-x = (t, t*m). So z_b - x_b = m*(z_a - x_a).
    # Note: x is NEVER in L(x, v) \ {x}.
    # So for a fixed x and z (z != x), z is in L(x, v) for EXACTLY ONE direction v?
    # Let's check.
    # If z_a != x_a, then for v=(0,1), z not in line.
    # For v=(1, m), we need m = (z_b - x_b) / (z_a - x_a). Since F_49 is a field, this m is unique.
    # So if z_a != x_a, z is in exactly ONE line through x (the one with slope m).
    # If z_a == x_a and z_b != x_b, then z is in the vertical line v=(0,1).
    # So for any z != x, z lies on exactly ONE line passing through x.
    # Let's call this direction v*(x, z).
    # Therefore, when we remove z, it affects exactly ONE direction for each x (where x != z).
    # Specifically, for each x != z, we increment cnt_not_in_N[x][v*(x, z)].
    # x becomes invalid if, after this increment, ALL directions v have cnt_not_in_N[x][v] > 0.
    # This happens if and only if:
    # 1. Before removal, x was valid (min cnt == 0).
    # 2. The set of directions with cnt == 0 for x was EXACTLY {v*(x, z)}.
    #    i.e., x had ONLY one valid direction, and that direction contained z.
    #    If x had another valid direction w != v*(x, z), then cnt_not_in_N[x][w] was 0 and remains 0, so x stays valid.
    
    # Algorithm:
    # 1. Initialize in_N = all True.
    # 2. cnt_not_in_N[x][v] = 0 for all x, v.
    # 3. valid_dir_count[x] = number of v such that cnt_not_in_N[x][v] == 0. Initially 50.
    # 4. Loop:
    #    a. Pick a point z (e.g., random or sequential) that is in N.
    #    b. Check if we can remove z.
    #       To remove z, we need to ensure no x becomes invalid.
    #       x becomes invalid iff valid_dir_count[x] == 1 AND the unique valid direction for x is v*(x, z).
    #       Let's precompute v*(x, z) for all pairs? No, too much memory (2401^2).
    #       But we can compute it on the fly for the check.
    #       Actually, we need to know for a given z, which x's are at risk.
    #       x is at risk if valid_dir_count[x] == 1.
    #       For those x, we need to check if the unique valid direction v_x contains z.
    #       If it does, removing z makes x invalid.
    #       If it doesn't, removing z is safe for x.
    #    c. If safe for all x, remove z (in_N[z] = False), update counts.
    #       Update: For all x != z, v = v*(x, z).
    #              cnt_not_in_N[x][v] += 1.
    #              if cnt_not_in_N[x][v] was 0, valid_dir_count[x] -= 1.
    #    d. Repeat until no more points can be removed or time is up.
    
    # Precompute v*(x, z) logic:
    # Given x=(xa, xb), z=(za, zb).
    # If xa == za: v = (0, 1) -> index 0.
    # Else: m = (zb - xb) * inv(za - xa). v = (1, m) -> index 1 + idx(m).
    
    # Precompute inverses
    inv = [0] * q
    for i in range(1, q):
        # Find inverse of idx_to_elem[i]
        e = idx_to_elem[i]
        for j in range(1, q):
            if f_mul(e, idx_to_elem[j]) == (1, 0):
                inv[i] = j
                break
    inv[0] = 0

    # State
    in_N = np.ones(total_points, dtype=bool)
    cnt_not_in_N = np.zeros((total_points, num_dirs), dtype=np.int32)
    valid_dir_count = np.full(total_points, num_dirs, dtype=np.int32)
    
    # Helper to get direction index from x, z
    def get_dir_idx(x, z):
        xa, xb = x
        za, zb = z
        if xa == za:
            return 0 # (0, 1)
        else:
            # m = (zb - xb) / (za - xa)
            num = f_add((zb, 0), f_add((0, 0), ((-xb) % p, 0))) # zb - xb
            # Wait, f_add takes pairs.
            num_pair = ((zb - xb) % p, 0)
            den_pair = ((za - xa) % p, 0)
            den_inv_idx = inv[elem_to_idx[den_pair]]
            m_idx = elem_to_idx[f_mul(num_pair, idx_to_elem[den_inv_idx])]
            return 1 + m_idx

    # To make the loop faster, we can maintain a list of "candidates" for removal.
    # Or just iterate through all points in N and try to remove them.
    # If we remove a point, we update the state.
    # We can do multiple passes.
    
    points_in_N = list(range(total_points))
    
    # Optimization: 
    # Instead of checking ALL x for each z, we can observe:
    # Removing z only affects x where valid_dir_count[x] == 1.
    # Let risky_xs = [x for x in range(total_points) if valid_dir_count[x] == 1]
    # If risky_xs is empty, we can remove ANY z in N.
    # If risky_xs is not empty, we can only remove z if for ALL x in risky_xs, 
    #    get_dir_idx(x, z) is NOT the unique valid direction of x.
    #    i.e., cnt_not_in_N[x][get_dir_idx(x, z)] > 0 is FALSE? 
    #    Wait. If valid_dir_count[x] == 1, there is exactly one v with cnt=0.
    #    Let v_unique(x) be that direction.
    #    x becomes invalid if v_unique(x) == get_dir_idx(x, z).
    #    So we must ensure v_unique(x) != get_dir_idx(x, z) for all risky x.
    
    # We need to know v_unique(x) for risky x.
    # We can store unique_dir[x] if valid_dir_count[x] == 1, else -1.
    
    unique_dir = np.full(total_points, -1, dtype=np.int32)
    
    def update_unique_dir(x):
        if valid_dir_count[x] == 1:
            for v in range(num_dirs):
                if cnt_not_in_N[x][v] == 0:
                    unique_dir[x] = v
                    return
        else:
            unique_dir[x] = -1

    # Initialize unique_dir (all have 50 valid dirs, so -1)
    
    # Main Loop
    # We will try to remove points. 
    # To be efficient, we can try to remove points that are "least critical".
    # Or just random/sequential.
    # Let's try sequential with a "no removal" counter to detect local optimum.
    
    no_removal_count = 0
    max_no_removal = 10000 # If we scan this many points without removing one, we might be stuck.
    # But 2401 points is small.
    
    # Let's use a random order for better exploration?
    # Or just iterate.
    
    while time.time() < deadline:
        removed_any = False
        # Shuffle order to avoid bias? 
        # For now, simple iteration.
        for z in range(total_points):
            if time.time() > deadline - 10:
                break
            if not in_N[z]:
                continue
            
            # Check if we can remove z
            can_remove = True
            
            # Find risky points: those with valid_dir_count == 1
            # We can't iterate all 2401 points for each z (2401^2 = 5.7M ops per pass).
            # 5.7M ops in Python is ~0.5s. 570s / 0.5s = 1000 passes.
            # Each pass removes at least 1 point (if we find one).
            # Max 2401 points. So ~2400 passes max. 2400 * 0.5s = 1200s. Too slow.
            
            # Optimization: Maintain a list of risky points.
            risky_points = [x for x in range(total_points) if valid_dir_count[x] == 1]
            
            if not risky_points:
                # Safe to remove z
                can_remove = True
            else:
                for x in risky_points:
                    # Get unique valid direction for x
                    v_u = unique_dir[x]
                    # Get direction from x to z
                    v_xz = get_dir_idx(idx_to_elem[x // q], idx_to_elem[x % q], 
                                       idx_to_elem[z // q], idx_to_elem[z % q])
                    if v_u == v_xz:
                        can_remove = False
                        break
            
            if can_remove:
                # Remove z
                in_N[z] = False
                removed_any = True
                
                # Update counts for all x != z
                # For each x, v = get_dir_idx(x, z)
                # cnt_not_in_N[x][v] += 1
                # if it was 0, valid_dir_count[x] -= 1
                # update unique_dir[x]
                
                for x in range(total_points):
                    if x == z:
                        continue
                    v_xz = get_dir_idx(idx_to_elem[x // q], idx_to_elem[x % q], 
                                       idx_to_elem[z // q], idx_to_elem[z % q])
                    if cnt_not_in_N[x, v_xz] == 0:
                        cnt_not_in_N[x, v_xz] = 1
                        valid_dir_count[x] -= 1
                        update_unique_dir(x)
                    else:
                        cnt_not_in_N[x, v_xz] += 1
                
                # If we removed a point, restart the loop or continue?
                # Restarting ensures we pick up new risky points correctly.
                # But we already have the list `risky_points` which is now stale.
                # So we should break out of the inner loop and restart the outer loop.
                break
        
        if not removed_any:
            # No point could be removed in this full pass.
            # We are at a local minimum (with respect to single-point removals).
            # To escape, we could try swapping? Or just stop.
            # Given the time budget, stopping is safer.
            # But maybe we can try a different order?
            # Let's just stop to be safe, as further search is complex.
            break
            
    # Build output
    points_list = []
    for pidx in range(total_points):
        if in_N[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])
            
    if not points_list:
        # Fallback: return all points
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])
            
    return np.array(points_list, dtype=np.int64)
