# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy:
    1. Start with the Hermitian unital baseline where the excluded set S
       consists of unital points whose y-coordinate has a non-zero square
       real part (inspiration 1's refinement).
    2. Greedily enlarge S: a point x can be added to S iff for every
       direction v, at least one other point on the line through x in
       direction v is already in S.
    3. Return N = F_49^2 \\ S.
    """
    p = 7
    q = p * p
    total = q * q
    deadline = time.time() + 580
    
    # alpha^2 = 3 (smallest quadratic non-residue mod 7)
    k_const = 3
    
    # Flat index: pt = (xa, xb, ya, yb) -> xa + xb*p + ya*p^2 + yb*p^3
    def pt_to_flat(xa, xb, ya, yb):
        return xa + xb * p + ya * p * p + yb * p * p * p
    
    def flat_to_pt(idx):
        xa = idx % p
        xb = (idx // p) % p
        ya = (idx // (p * p)) % p
        yb = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))
    directions.append((0, 1, 0, 0))
    n_dirs = len(directions)
    
    # Precompute line_id for each point in each direction
    line_id = np.zeros((total, n_dirs), dtype=np.int32)
    
    for d in range(n_dirs):
        vxa, vxb, vya, vyb = directions[d]
        for pt_idx in range(total):
            xa, xb, ya, yb = flat_to_pt(pt_idx)
            if vxa == 0:  # v = (0, 1)
                lid = xa + xb * p
            else:
                # v = (1, m) where m = (vya, vyb)
                mxa = (vya * xa + vyb * xb * k_const) % p
                mxb = (vya * xb + vyb * xa) % p
                lda = (ya - mxa) % p
                ldb = (yb - mxb) % p
                lid = lda + ldb * p
            line_id[pt_idx, d] = lid
    
    # Build line_points: for each direction, map line_id -> list of point indices
    line_points = [dict() for _ in range(n_dirs)]
    for d in range(n_dirs):
        for pt_idx in range(total):
            lid = line_id[pt_idx, d]
            if lid not in line_points[d]:
                line_points[d][lid] = []
            line_points[d][lid].append(pt_idx)
    
    # Baseline: Hermitian unital with refined special set
    # U = {(x,y) : 2*xa + ya^2 - yb^2*k = 0 mod p}
    # Y_special = {(u^2 mod p, v) : u in 1..6, v in 0..6}  (non-zero squares only)
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    S_mask = np.zeros(total, dtype=bool)
    for pt_idx in range(total):
        xa, xb, ya, yb = flat_to_pt(pt_idx)
        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            S_mask[pt_idx] = True
    
    # Coverage: for each point and direction, count how many OTHER points on the
    # line are in S
    coverage = np.zeros((total, n_dirs), dtype=np.int32)
    
    def update_coverage(pt_idx, add=True):
        sign = 1 if add else -1
        for d in range(n_dirs):
            lid = line_id[pt_idx, d]
            for other_idx in line_points[d][lid]:
                if other_idx != pt_idx:
                    coverage[other_idx, d] += sign
    
    # Initialize coverage with baseline S
    for pt_idx in range(total):
        if S_mask[pt_idx]:
            update_coverage(pt_idx, add=True)
    
    # Greedily add points to S
    # A point can be added if min_d coverage[pt][d] >= 1
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5:
            break
        
        min_cov = np.min(coverage, axis=1)
        candidates = np.where((~S_mask) & (min_cov >= 1))[0]
        
        if len(candidates) == 0:
            break
        
        for pt_idx in candidates:
            S_mask[pt_idx] = True
            update_coverage(int(pt_idx), add=True)
        
        changed = True
    
    # Build N = all points NOT in S
    N_indices = [i for i in range(total) if not S_mask[i]]
    
    points = np.array([[[flat_to_pt(i)[0], flat_to_pt(i)[1]],
                        [flat_to_pt(i)[2], flat_to_pt(i)[3]]] for i in N_indices],
                      dtype=np.int64)
    
    return points

# EVOLVE-BLOCK-END
