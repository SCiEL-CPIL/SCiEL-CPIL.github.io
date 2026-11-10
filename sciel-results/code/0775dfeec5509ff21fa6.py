import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49  # field size
    # alpha^2 = 3
    alpha2 = 3
    
    # Encode field element (a,b) -> a + 7*b (0..48)
    # x = xa + xb*alpha, so x_idx = xa + 7*xb
    # y = ya + yb*alpha, y_idx = ya + 7*yb
    # point index = x_idx * 49 + y_idx (0..2400)
    
    # Field operations on encoded elements
    # (a,b) + (c,d) = (a+c, b+d) mod 7
    def fe_add(x, y):
        xa, xb = x % 7, x // 7
        ya, yb = y % 7, y // 7
        return ((xa+ya)%7) + 7*((xb+yb)%7)
    def fe_mul(x, y):
        xa, xb = x % 7, x // 7
        ya, yb = y % 7, y // 7
        # (xa + xb*alpha)(ya + yb*alpha) = xa*ya + (xa*yb + xb*ya)*alpha + xb*yb*alpha^2
        # = xa*ya + xb*yb*3 + (xa*yb + xb*ya)*alpha
        a = (xa*ya + xb*yb*alpha2) % 7
        b = (xa*yb + xb*ya) % 7
        return a + 7*b
    
    # Directions: 49 slopes m (v=(1,m)) + 1 vertical (v=(0,1))
    # For direction d (0..48): v = (1, m) where m = d
    # For direction 49: v = (0, 1)
    
    # Precompute for each direction d and each point, the line membership.
    # line_id[d][point] and lines[d][line_id] = list of 49 points.
    
    # For d in 0..48 (slope m=d): line through (x,y) has intercept c = y - x*m.
    #   line_id = c. Points on line c: {(x, x*m + c) for x in 0..48}.
    # For d=49 (vertical): line_id = x. Points: {(x, y) for y in 0..48}.
    
    num_dirs = 50
    # lines[d] is a dict or array mapping line_id -> list of point indices
    # We'll store lines[d] as a 2D array of shape (49, 49): lines[d][line_id][t] = point index
    lines = np.zeros((num_dirs, 49, 49), dtype=np.int32)
    line_id = np.zeros((num_dirs, 2401), dtype=np.int32)
    
    for d in range(49):
        m = d
        for x_idx in range(49):
            for y_idx in range(49):
                # intercept c = y - x*m
                c = fe_add(y_idx, fe_add(0, 0))  # placeholder
                # c = y_idx - x_idx * m
                xm = fe_mul(x_idx, m)
                c = fe_add(y_idx, 49 - (xm % 49))  # y - xm; but fe elements are 0..48, negation is (49 - xm) % 49
                # Actually fe_add(y_idx, -xm) where -xm = (49 - xm) % 49
                c = (y_idx - xm) % 49
                line_id[d, x_idx*49 + y_idx] = c
        for c in range(49):
            for t in range(49):
                x_idx = t  # x = t
                y_idx = fe_add(fe_mul(x_idx, m), c)
                lines[d, c, t] = x_idx*49 + y_idx
    
    # vertical direction d=49
    for x_idx in range(49):
        for y_idx in range(49):
            line_id[49, x_idx*49 + y_idx] = x_idx
    for x_idx in range(49):
        for t in range(49):
            y_idx = t
            lines[49, x_idx, t] = x_idx*49 + y_idx
    
    # For each point p, precompute the affected (x, d) pairs: for each d, the 48 other points on p's d-line.
    # affected[p] = flat array of 2400 entries, each = x*50 + d (index into block.ravel())
    # affected_x[p] = flat array of 2400 x values
    affected = np.zeros((2401, 2400), dtype=np.int32)
    affected_x = np.zeros((2401, 2400), dtype=np.int32)
    for p in range(2401):
        x_idx = p // 49
        y_idx = p % 49
        slot = 0
        for d in range(50):
            lid = line_id[d, p]
            for t in range(49):
                q_pt = lines[d, lid, t]
                if q_pt == p:
                    continue
                affected[p, slot] = q_pt * 50 + d
                affected_x[p, slot] = q_pt
                slot += 1
        assert slot == 2400
    
    # Now the greedy grow.
    def build_block(C_set):
        # C_set: set of point indices in C
        block = np.zeros((2401, 50), dtype=np.int32)
        for p in C_set:
            block_flat = block.ravel()
            idxs = affected[p]
            block_flat[idxs] += 1
        cover_count = (block == 0).sum(axis=1)
        return block, cover_count
    
    def try_add(block, cover_count, p):
        # Returns (valid, new_block, new_cover_count) or we do in-place with rollback
        block_flat = block.ravel()
        idxs = affected[p]
        old_vals = block_flat[idxs].copy()
        block_flat[idxs] += 1
        # compute cover_count changes
        xs = affected_x[p]
        # for each idx where old_val==0, cover_count[x] decreases by 1
        zero_mask = (old_vals == 0)
        if not np.any(zero_mask):
            return True, None, None  # no cover changes, valid
        xs_zero = xs[zero_mask]
        uniq, counts = np.unique(xs_zero, return_counts=True)
        new_cc = cover_count[uniq] - counts
        if np.any(new_cc <= 0):
            # invalid, rollback
            block_flat[idxs] = old_vals
            return False, None, None
        # commit
        cover_count[uniq] = new_cc
        return True, None, None
    
    # Greedy grow from a starting C
    def greedy_grow(start_C, order, deadline):
        C = set(start_C)
        block, cover_count = build_block(C)
        # candidates = points not in C
        # try to add in 'order'
        added = 0
        for p in order:
            if time.time() > deadline:
                break
            if p in C:
                continue
            valid, _, _ = try_add(block, cover_count, p)
            if valid:
                C.add(p)
                added += 1
        return C, block, cover_count
    
    # Build the seed C
    k = 3  # alpha^2
    y_special = set()
    for u in range(7):
        for v in range(7):
            y_special.add(((u*u)%7, v))
    seed_C = set()
    for xa in range(7):
        for xb in range(7):
            for ya in range(7):
                for yb in range(7):
                    x_idx = xa + 7*xb
                    y_idx = ya + 7*yb
                    pt = x_idx*49 + y_idx
                    in_unital = (2*xa + ya*ya - yb*yb*k) % 7 == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        seed_C.add(pt)
    
    # Random restarts
    best_C = seed_C
    deadline = time.time() + 600 - 30  # 30s safety margin
    
    # Also try from empty
    rng = np.random.default_rng(0)
    all_points = np.arange(2401)
    
    starts = [seed_C, set()]
    for start in starts:
        for restart in range(200):
            if time.time() > deadline:
                break
            order = rng.permutation(all_points)
            C, block, cc = greedy_grow(start, order, deadline)
            if len(C) > len(best_C):
                best_C = C
    
    # Build N = all points \ best_C
    C_arr = np.array(sorted(best_C), dtype=np.int64)
    N_mask = np.ones(2401, dtype=bool)
    N_mask[C_arr] = False
    N_pts = np.where(N_mask)[0]
    
    # Convert to [[xa,xb],[ya,yb]]
    result = np.zeros((len(N_pts), 2, 2), dtype=np.int64)
    for i, pt in enumerate(N_pts):
        x_idx = pt // 49
        y_idx = pt % 49
        xa, xb = x_idx % 7, x_idx // 7
        ya, yb = y_idx % 7, y_idx // 7
        result[i] = [[xa, xb], [ya, yb]]
    
    return result
