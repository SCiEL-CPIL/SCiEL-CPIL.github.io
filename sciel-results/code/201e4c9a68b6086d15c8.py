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
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    zero_counts = np.zeros(NPTS, dtype=np.int32)

    def recompute(S_list):
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
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    def apply_remove(p_idx):
        # Returns list of changes to revert if needed, or None if it would break validity?
        # We don't use remove in the main greedy, but might for swaps.
        # For now, just update counts. Caller must check validity.
        changes = []
        for d_idx in range(num_dirs):
            for n_idx in affected[p_idx, d_idx]:
                if line_count[n_idx, d_idx] == 1:
                    changes.append((n_idx, d_idx))
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] -= 1
            zero_counts[n_idx] += 1
        return changes

    def revert_remove(p_idx, changes):
        for n_idx, d_idx in changes:
            line_count[n_idx, d_idx] += 1
            zero_counts[n_idx] -= 1

    # --- Greedy Growth ---
    np.random.seed(42)
    best_S = set()
    best_size = 0

    # Try multiple random seeds (single points)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        # Seed with a random point
        seed_idx = np.random.randint(NPTS)
        S = {seed_idx}
        recompute(S)
        
        if not is_valid():
            continue

        outside = [i for i in range(NPTS) if i not in S]
        
        # Greedy pass
        while time.time() < deadline - 10:
            np.random.shuffle(outside)
            added_any = False
            for p_idx in outside:
                if time.time() > deadline - 10:
                    break
                if p_idx in S:
                    continue
                changes = try_add(p_idx)
                if changes is not None:
                    apply_add(p_idx, changes)
                    S.add(p_idx)
                    outside.remove(p_idx)
                    added_any = True
            
            if not added_any:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            # Optional: Try to continue improving best_S directly?
            # No, keep it simple.

    # If best_S is still empty (shouldn't happen), use {0}
    if not best_S:
        best_S = {0}
        recompute(best_S)

    # Final check and construction
    # Note: recompute was done inside the loop for the last seed.
    # But if we broke early, we need to ensure state matches best_S.
    # Actually, the loop updates S and counts together.
    # If we switched to a new seed, we recomputed.
    # So current state corresponds to the last processed seed.
    # We need to make sure best_S is the one we want.
    # The loop keeps best_S as a set.
    # We should recompute best_S to be safe before returning?
    # No, we just need to return the complement.
    
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
