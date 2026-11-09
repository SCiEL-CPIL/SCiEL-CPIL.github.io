import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50 directions: v=(0,1) and v=(1,m) for m in F_q
    k_const = 3

    # Deadline guard - be conservative
    deadline = time.time() + 560.0

    # Map (a, b) -> index a*p + b
    idx_to_elem = [(i // p, i % p) for i in range(q)]

    # Precompute multiplication table for F_q
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    # Direction setup
    # dir_idx 0 -> (0, 1) where 1 is the multiplicative identity (index 7)
    # dir_idx i (1<=i<=49) -> (1, i-1) where 1 is index 7
    dir_dx = np.zeros(num_dirs, dtype=np.int32)
    dir_dy = np.zeros(num_dirs, dtype=np.int32)
    dir_dx[0] = 0
    dir_dy[0] = 7  # field element 1
    for i in range(1, num_dirs):
        dir_dx[i] = 7  # field element 1
        dir_dy[i] = i - 1

    # Precompute line membership
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(q, dtype=np.int32)

    line_members = np.zeros((total_points, num_dirs, q), dtype=np.int32)
    for v_idx in range(num_dirs):
        dx = dir_dx[v_idx]
        dy = dir_dy[v_idx]
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        line_members[:, v_idx, :] = nx * q + ny

    # For each point z, precompute affected (x, v) pairs
    # When z is in N, it "blocks" direction v for all x on line(z,v) \ {z}
    # So affected pairs are (x, v) where x is on line through z in direction v, x != z
    affected_x = []
    affected_v = []
    for z in range(total_points):
        xs = []
        vs = []
        for v in range(num_dirs):
            other_pts = line_members[z, v, 1:]  # exclude t=0 which is z itself
            xs.extend(other_pts.tolist())
            vs.extend([v] * len(other_pts))
        affected_x.append(np.array(xs, dtype=np.int32))
        affected_v.append(np.array(vs, dtype=np.int32))

    # Strategy: Start with N = all points (valid trivially? No!)
    # Actually, N = all points means for each x, line(x,v)\{x} subset N for all v, so valid.
    # But |N| = 2401, complement = 0. We want to maximize complement.
    # 
    # Better: Start with N empty, then greedily add points to N while maintaining validity.
    # Actually, the condition is: for EVERY x in F_q^2, there exists a direction v such that
    # line(x, v) \ {x} is ALL in N.
    #
    # So N must be "dense enough" that every point is covered by at least one full line (minus itself).
    #
    # Let's think differently. Define:
    # covered[x] = True if exists v such that line(x,v)\{x} subset N
    # We need covered[x] = True for all x.
    #
    # Start with N = {} (empty). Then no x is covered (unless q=1, but q=49).
    # As we add points to N, we may cover more x's.
    #
    # Actually this is the wrong direction. We want to MINIMIZE |N| while maintaining validity.
    # Equivalently, MAXIMIZE |complement| = 2401 - |N|.
    #
    # Let E = complement of N. We want to maximize |E| such that for every x,
    # there exists v with line(x,v)\{x} subset N, i.e., line(x,v)\{x} intersect E = empty.
    #
    # So: for every x, exists v such that no point of E (other than possibly x itself) lies on line(x,v).
    # Wait, x may or may not be in E. If x in E, then x is not in N, but that's fine -
    # the condition is about line(x,v)\{x}, not including x.
    #
    # So the condition is: for every x, exists v such that (line(x,v)\{x}) ∩ E = ∅.
    # I.e., no point of E (other than x) is on that line.
    #
    # Let's reframe: For each x, define bad_dirs(x) = {v : (line(x,v)\{x}) ∩ E ≠ ∅}
    # We need bad_dirs(x) ≠ all dirs, i.e., |bad_dirs(x)| < 50 for all x.
    #
    # When we add z to E, it increases bad_dirs for all x such that z ∈ line(x,v)\{x} for some v.
    # I.e., x ∈ line(z,v)\{z} for some v. So x is on some line through z, x ≠ z.
    #
    # This is exactly the affected pairs computation above.
    #
    # Greedy approach:
    # Start with E = {}. For each candidate z (in some order), try to add z to E.
    # Check if this causes any x to have all 50 directions blocked.
    # If not, add z. If yes, skip z.
    #
    # Multiple restarts with different orders.
    #
    # But this greedy might be suboptimal. Let's also try local search.
    
    def try_add_to_E(z, in_E, blocked_counts):
        """Try to add z to E. Return True if successful, False if it would violate constraints."""
        ax = affected_x[z]
        av = affected_v[z]
        
        # Apply
        blocked_counts[ax, av] += 1
        in_E[z] = True
        
        # Check if any x now has all directions blocked
        unique_affected = np.unique(ax)
        min_blocked = np.min(blocked_counts[unique_affected, :], axis=1)
        
        if np.any(min_blocked >= num_dirs):
            # Revert
            blocked_counts[ax, av] -= 1
            in_E[z] = False
            return False
        return True

    def remove_from_E(z, in_E, blocked_counts):
        """Remove z from E. Always valid (removing can't cause violations)."""
        ax = affected_x[z]
        av = affected_v[z]
        blocked_counts[ax, av] -= 1
        in_E[z] = False

    best_E = None
    best_complement = 0
    rng = random.Random(42)
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        in_E = np.zeros(total_points, dtype=bool)
        blocked_counts = np.zeros((total_points, num_dirs), dtype=np.int32)
        
        # Random order
        order = list(range(total_points))
        rng.shuffle(order)
        
        size_E = 0
        
        # Phase 1: Greedy add
        for z in order:
            if time.time() > deadline:
                break
            if in_E[z]:
                continue
            if try_add_to_E(z, in_E, blocked_counts):
                size_E += 1
        
        # Phase 2: Local search - try to add more points that were skipped
        # Also try swaps: remove one point, add another
        improved = True
        swap_iterations = 0
        while improved and swap_iterations < 200:
            improved = False
            swap_iterations += 1
            
            if time.time() > deadline:
                break
            
            # Try adding skipped points
            for z in order:
                if in_E[z]:
                    continue
                if try_add_to_E(z, in_E, blocked_counts):
                    size_E += 1
                    improved = True
            
            # Try swaps: for each z in E, try replacing it with a z' not in E
            e_indices = np.where(in_E)[0]
            if len(e_indices) == 0:
                break
                
            # Sample a few swaps
            n_swaps_to_try = min(50, len(e_indices))
            for _ in range(n_swaps_to_try):
                if time.time() > deadline:
                    break
                z_out = e_indices[rng.randint(0, len(e_indices) - 1)]
                # Pick random z_in not in E
                while True:
                    z_in = rng.randint(0, total_points - 1)
                    if not in_E[z_in]:
                        break
                
                # Try the swap
                ax_out = affected_x[z_out]
                av_out = affected_v[z_out]
                ax_in = affected_x[z_in]
                av_in = affected_v[z_in]
                
                # Apply swap
                blocked_counts[ax_out, av_out] -= 1
                in_E[z_out] = False
                blocked_counts[ax_in, av_in] += 1
                in_E[z_in] = True
                
                # Check validity
                unique_affected = np.unique(np.concatenate([ax_out, ax_in]))
                min_blocked = np.min(blocked_counts[unique_affected, :], axis=1)
                
                if np.any(min_blocked >= num_dirs):
                    # Revert
                    blocked_counts[ax_out, av_out] += 1
                    in_E[z_out] = True
                    blocked_counts[ax_in, av_in] -= 1
                    in_E[z_in] = False
                else:
                    # Keep swap
                    # size_E unchanged
                    improved = True
                    # Update e_indices (roughly)
                    e_indices = np.where(in_E)[0]
        
        if size_E > best_complement:
            best_complement = size_E
            best_E = in_E.copy()
    
    # Build N = complement of E
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
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
