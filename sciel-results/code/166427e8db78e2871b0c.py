# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_q^2 with q = p^2, p = 7.
    Maximize |N^c| = |S|.
    """
    p = 7
    q = p * p
    NPTS = q * q  # 2401

    start = time.time()
    deadline = start + 580.0  # safety margin

    # --- Field F_p[alpha], alpha^2 = k ---
    k_const = 3  # 3 is a non-residue mod 7 (3^3 = 27 = 6 = -1)

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # --- Directions: (1, m) for all m in F_q, plus (0, 1) ---
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # --- Precompute affected[p, d, t] ---
    affected = np.zeros((NPTS, num_dirs, q - 1), dtype=np.int32)
    for p_idx in range(NPTS):
        px, py = idx_to_pt(p_idx)[0], idx_to_pt(p_idx)[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    affected[p_idx, d_idx, count] = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    count += 1

    # --- State ---
    in_S = np.zeros(NPTS, dtype=np.int32)
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    zero_counts = np.zeros(NPTS, dtype=np.int32)

    def recompute(S_list):
        in_S[:] = 0
        for s_idx in S_list:
            in_S[s_idx] = 1
        line_count[:] = 0
        for s_idx in S_list:
            for d_idx in range(num_dirs):
                line_count[affected[s_idx, d_idx], d_idx] += 1
        zero_counts[:] = np.sum(line_count == 0, axis=1)

    def is_valid():
        return bool(np.all(zero_counts > 0))

    def try_add(p_idx):
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 0:
                    if zero_counts[n_idx] == 1:
                        return None
                    changes.append((n_idx, d_idx))
        return changes

    def apply_add(p_idx, changes):
        in_S[p_idx] = 1
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    def try_remove(p_idx):
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 1:
                    changes.append((n_idx, d_idx))
        return changes

    def apply_remove(p_idx, changes):
        in_S[p_idx] = 0
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] -= 1
            zero_counts[n_idx] += 1

    def revert_remove(p_idx, changes):
        in_S[p_idx] = 1
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    # --- Greedy Growth with Swap-based Local Search ---
    np.random.seed(42)
    best_S = set()
    best_size = 0

    # Try multiple random seeds
    for trial in range(8):
        if time.time() > deadline - 30:
            break
        
        # Seed with a random point
        seed_idx = np.random.randint(NPTS)
        S = {seed_idx}
        recompute(S)
        
        if not is_valid():
            continue

        outside = set(range(NPTS)) - S
        
        # Greedy pass
        while time.time() < deadline - 30:
            outside_list = list(outside)
            np.random.shuffle(outside_list)
            added_any = False
            for p_idx in outside_list:
                if time.time() > deadline - 30:
                    break
                if p_idx in S:
                    continue
                changes = try_add(p_idx)
                if changes is not None:
                    apply_add(p_idx, changes)
                    S.add(p_idx)
                    outside.discard(p_idx)
                    added_any = True
            
            if not added_any:
                break
        
        # Swap-based local search: try to replace one point with another to increase size
        if time.time() < deadline - 30:
            S_list = list(S)
            np.random.shuffle(S_list)
            improved = True
            while improved and time.time() < deadline - 30:
                improved = False
                for p_idx in S_list:
                    if time.time() > deadline - 30:
                        break
                    # Try removing p_idx
                    rem_changes = try_remove(p_idx)
                    apply_remove(p_idx, rem_changes)
                    
                    # Try adding a random point from outside
                    outside_candidates = list(outside)
                    np.random.shuffle(outside_candidates)
                    for new_idx in outside_candidates[:50]:  # Try up to 50 candidates
                        if time.time() > deadline - 30:
                            break
                        add_changes = try_add(new_idx)
                        if add_changes is not None:
                            # This swap is valid
                            apply_add(new_idx, add_changes)
                            S.discard(p_idx)
                            S.add(new_idx)
                            outside.discard(new_idx)
                            outside.add(p_idx)
                            improved = True
                            break
                    
                    # Revert removal if no swap was made
                    if not improved:
                        revert_remove(p_idx, rem_changes)
            
            # Update best
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    # If best_S is still empty, use {0}
    if not best_S:
        best_S = {0}
        recompute(best_S)

    # Final check and construction
    S_final = best_S
    arr = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx([[xa, xb], [ya, yb]])
                    if idx not in S_final:
                        arr.append([[xa % p, xb % p], [ya % p, yb % p]])
    
    return np.array(arr, dtype=np.int64)


# EVOLVE-BLOCK-END
