# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Nikodym set in F_49^2, F_49 = F_7[alpha], alpha^2 = 3.

    Strategy:
      1. Start from the baseline construction (Hermitian unital with y_a square
         excluded), which is known to be a valid Nikodym set.
      2. Use Simulated Annealing to enlarge the complement S (shrink N).
         - State: S_mask (points in S).
         - Objective: Maximize |S| subject to validity (every point x in F_49^2 
           must have at least one mate in S for every direction v).
         - Moves: Add a point to S or remove a point from S.
         - Efficiency: Maintain coverage counts to quickly check validity and 
           calculate energy changes.
    """
    p = 7
    q = p * p  # 49
    total = q * q  # 2401

    # alpha^2 = 3 (smallest quadratic non-residue mod 7)
    k_const = 3

    deadline = time.time() + 580  # safety margin before 630s kill

    # ---------- field arithmetic (vectorized over arrays of (a, b) pairs) ----------
    def add_f(A, B):
        """A, B: arrays of shape (..., 2); returns (A+B) mod p componentwise."""
        return (A + B) % p

    def mul_f(A, B):
        """
        Multiply two arrays of field elements.
        A, B: shape (..., 2), representing (a, b) = a + b*alpha.
        (a1 + b1*alpha)(a2 + b2*alpha) = (a1*a2 + b1*b2*k) + (a1*b2 + b1*a2)*alpha
        """
        a1, b1 = A[..., 0], A[..., 1]
        a2, b2 = B[..., 0], B[..., 1]
        real = (a1 * a2 + b1 * b2 * k_const) % p
        imag = (a1 * b2 + b1 * a2) % p
        return np.stack([real, imag], axis=-1)

    # ---------- flat-index helpers ----------
    # Point (xa, xb, ya, yb) -> flat index = xa * 343 + xb * 49 + ya * 7 + yb
    def to_flat(pt):
        """pt: shape (..., 4) array of (xa, xb, ya, yb) ints."""
        return (pt[..., 0] * 343 + pt[..., 1] * 49 + pt[..., 2] * 7 + pt[..., 3])

    def from_flat(idx):
        """idx: array of flat indices -> shape (..., 4) array."""
        idx = idx.astype(np.int64)
        xa = (idx // 343) % p
        xb = (idx // 49) % p
        ya = (idx // 7) % p
        yb = idx % p
        return np.stack([xa, xb, ya, yb], axis=-1)

    # Build flat index for every point
    all_flat = np.arange(total, dtype=np.int64)
    all_pts = from_flat(all_flat)  # shape (2401, 4)

    # ---------- baseline complement S ----------
    # S = { (xa, xb, ya, yb) : 2*xa + ya^2 - yb^2*k = 0 mod p  AND  ya^2 != ya mod p }
    # i.e., ya is NOT a quadratic residue (including 0? 0 is a residue: 0=0^2)
    # Squares mod 7: {0, 1, 2, 4}. Non-squares: {3, 5, 6}.
    # Baseline excludes points on the unital whose ya is a non-square.
    ya_arr = all_pts[:, 2]
    yb_arr = all_pts[:, 3]
    xa_arr = all_pts[:, 0]

    # in_unital: 2*xa + ya^2 - yb^2 * k_const == 0 mod 7
    in_unital = (2 * xa_arr + ya_arr * ya_arr - yb_arr * yb_arr * k_const) % p == 0

    # ya is a square mod 7?
    squares_mod7 = {0, 1, 2, 4}
    ya_is_square = np.array([ya in squares_mod7 for ya in ya_arr])

    # S = unital points with ya NOT a square
    S_mask = np.zeros(total, dtype=bool)
    S_mask[in_unital & ~ya_is_square] = True

    # ---------- directions ----------
    # v = (1, m) for m in F_49  ->  49 directions
    # v = (0, 1)                ->   1 direction
    # Total: 50 directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 1, 0, 0))
    n_dirs = len(dirs)  # 50

    # ---------- precompute line mates ----------
    # For each point x (flat index) and each direction v,
    # compute the 48 other points on the line {x + t*v : t in F_49, t != 0}.
    # Store as line_mates[point_idx, dir_idx, j] = flat index of j-th mate.
    # Shape: (2401, 50, 48)

    # All non-zero t in F_49: 48 elements
    t_vals = []
    for ta in range(p):
        for tb in range(p):
            if ta == 0 and tb == 0:
                continue
            t_vals.append((ta, tb))
    t_arr = np.array(t_vals, dtype=np.int64)  # shape (48, 2)

    line_mates = np.zeros((total, n_dirs, 48), dtype=np.int64)

    for di, v in enumerate(dirs):
        vxa, vxb, vya, vyb = v
        ta = t_arr[:, 0]
        tb = t_arr[:, 1]
        tv_xa = (ta * vxa + tb * vxb * k_const) % p
        tv_xb = (ta * vxb + tb * vxa) % p
        tv_ya = (ta * vya + tb * vyb * k_const) % p
        tv_yb = (ta * vyb + tb * vya) % p
        # Compute flat indices of x + tv
        x_all = all_pts[:, np.newaxis, :]  # (2401, 1, 4)
        tv_all = np.stack([tv_xa, tv_xb, tv_ya, tv_yb], axis=-1)  # (48, 4)
        sum_pts = (x_all + tv_all[np.newaxis, :, :]) % p  # (2401, 48, 4)
        flat_mates = to_flat(sum_pts)  # (2401, 48)
        line_mates[:, di, :] = flat_mates

    # ---------- Simulated Annealing ----------
    # State: S_mask (bool array of size 2401)
    # Coverage: count[x, v] = number of mates of x in direction v that are in S.
    # Validity condition: min_v count[x, v] >= 1 for all x.
    # Energy: number of (x, v) pairs where count[x, v] == 0.
    # We want to minimize Energy (ideally 0) while maximizing |S|.
    
    # Initialize coverage counts
    # count[x, v] = sum(S_mask[line_mates[x, v, :]])
    # Vectorized: S_mask[line_mates] is (2401, 50, 48) bool. Sum over last axis.
    coverage = S_mask[line_mates].sum(axis=-1)  # shape (2401, 50)
    
    # Current energy
    energy = np.sum(coverage == 0)
    
    # Best state
    best_S_mask = S_mask.copy()
    best_energy = energy
    best_complement_size = np.sum(S_mask)
    
    T = 10.0
    cooling = 0.995
    T_min = 0.01
    
    # Precompute list of points for random sampling
    all_indices = np.arange(total)
    
    # Precompute direction indices for vectorized updates if needed, 
    # but simple loop is likely faster due to memory locality or just low overhead.
    
    # Let's use the simple loop for clarity and correctness first.
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        
        # Propose move
        # To maximize complement, we should bias towards adding points.
        # But we must be able to remove to escape bad states.
        # Let's say 70% add, 30% remove.
        move_type = 0 if np.random.random() < 0.7 else 1
        pt_idx = np.random.choice(all_indices)
        
        delta_E = 0
        
        if move_type == 0:
            # Add pt_idx to S
            if S_mask[pt_idx]:
                continue
            S_mask[pt_idx] = True
            # Update coverage
            # For each direction v, the 48 mates of pt_idx gain +1 coverage.
            # We need to track how many of these mates had 0 coverage before.
            # If a mate had 0, and we add 1, it becomes 1. Energy decreases by 1 for that (mate, v) pair.
            # If a mate had >0, energy doesn't change.
            
            # We need to count how many (mate, v) pairs had coverage 0.
            # Let M = line_mates[pt_idx] (50, 48)
            # For each v, check coverage[M[v], v] == 0.
            
            # Vectorized check:
            # We want to sum over all v and mates j: (coverage[M[v,j], v] == 0)
            # This is tricky to vectorize directly on the 2D coverage array with varying column indices.
            
            # Let's do it in a loop over directions.
            # 50 iterations.
            old_energy = energy
            for v in range(n_dirs):
                mates = line_mates[pt_idx, v]
                # Get current coverage for these mates in direction v
                cov_vals = coverage[mates, v]
                # Count zeros
                zeros = np.sum(cov_vals == 0)
                # Update coverage
                coverage[mates, v] += 1
                # Energy change: -zeros (since 0->1 reduces violations)
                # Wait, is it that simple?
                # Yes, if coverage was 0, it was a violation. Now it is 1, so not a violation.
                # If coverage was >0, it was not a violation. Now it is >1, still not a violation.
                # So delta_E for this direction is -zeros.
                # But we need total delta_E.
            
            # Recalculating energy from scratch is O(2401*50) = 120k ops.
            # Updating incrementally is O(50*48) = 2400 ops.
            # Incremental is much better.
            
            # Let's redo the loop to accumulate delta_E.
            S_mask[pt_idx] = False # Rollback to calculate delta first? 
            # Or calculate delta, apply, update energy.
            
            # Let's calculate delta_E first.
            delta_E = 0
            for v in range(n_dirs):
                mates = line_mates[pt_idx, v]
                cov_vals = coverage[mates, v]
                zeros = np.sum(cov_vals == 0)
                delta_E -= zeros
            
            # Apply move
            S_mask[pt_idx] = True
            for v in range(n_dirs):
                mates = line_mates[pt_idx, v]
                coverage[mates, v] += 1
            
            energy += delta_E
            
        else:
            # Remove pt_idx from S
            if not S_mask[pt_idx]:
                continue
            S_mask[pt_idx] = False
            # Update coverage
            # For each direction v, the 48 mates of pt_idx lose 1 coverage.
            # If a mate had 1, it becomes 0. Energy increases by 1 for that (mate, v) pair.
            # If a mate had >1, it becomes >0. Energy doesn't change.
            
            delta_E = 0
            for v in range(n_dirs):
                mates = line_mates[pt_idx, v]
                cov_vals = coverage[mates, v]
                ones = np.sum(cov_vals == 1)
                delta_E += ones
            
            # Apply move
            S_mask[pt_idx] = False
            for v in range(n_dirs):
                mates = line_mates[pt_idx, v]
                coverage[mates, v] -= 1
            
            energy += delta_E
            
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            # Boltzmann probability
            # Avoid overflow
            if T > 0:
                prob = np.exp(-delta_E / T)
            else:
                prob = 0
            accept = np.random.random() < prob
        
        if not accept:
            # Rollback
            if move_type == 0:
                S_mask[pt_idx] = False
                for v in range(n_dirs):
                    mates = line_mates[pt_idx, v]
                    coverage[mates, v] -= 1
                energy -= delta_E
            else:
                S_mask[pt_idx] = True
                for v in range(n_dirs):
                    mates = line_mates[pt_idx, v]
                    coverage[mates, v] += 1
                energy -= delta_E
            
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_S_mask = S_mask.copy()
            best_complement_size = np.sum(S_mask)
        elif energy == best_energy and np.sum(S_mask) > best_complement_size:
            # Same validity, larger complement
            best_energy = energy
            best_S_mask = S_mask.copy()
            best_complement_size = np.sum(S_mask)
            
        # Cool down
        T *= cooling
        
    # Use the best state found
    S_mask = best_S_mask
    
    # ---------- build output ----------
    # N = all points not in S
    N_mask = ~S_mask
    N_pts = all_pts[N_mask]  # shape (k, 4)
    k = len(N_pts)

    # Convert to (k, 2, 2) array: [[xa, xb], [ya, yb]]
    if k == 0:
        # Should not happen, but guard against it
        points = np.zeros((1, 2, 2), dtype=np.int64)
    else:
        points = np.empty((k, 2, 2), dtype=np.int64)
        points[:, 0, 0] = N_pts[:, 0]
        points[:, 0, 1] = N_pts[:, 1]
        points[:, 1, 0] = N_pts[:, 2]
        points[:, 1, 1] = N_pts[:, 3]

    return points


# EVOLVE-BLOCK-END
