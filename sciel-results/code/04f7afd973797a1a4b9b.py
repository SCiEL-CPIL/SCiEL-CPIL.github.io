import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    
    def mul_pt(p1, p2):
        return ((p1[0]*p2[0] + 3*p1[1]*p2[1]) % p, (p1[0]*p2[1] + p1[1]*p2[0]) % p)
        
    def to_coords(idx):
        return (idx // q, idx % q)

    elems = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(elems)}
    
    inv = {}
    for e in elems:
        if e == (0,0): continue
        for f in elems:
            if f == (0,0): continue
            if mul_pt(e, f) == (1, 0):
                inv[e] = f
                break
                
    D = np.zeros((n_points, n_points), dtype=np.int8)
    for y_idx in range(n_points):
        y = to_coords(y_idx)
        for x_idx in range(n_points):
            if x_idx == y_idx: continue
            x = to_coords(x_idx)
            dx = ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
            if dx == (0, 0): continue
            if dx[0] == 0:
                D[y_idx, x_idx] = 49
            else:
                dy = dx[1]
                m = mul_pt(dy, inv[dx])
                D[y_idx, x_idx] = elem_to_idx[m]
                
    # Precompute line masks?
    # lines[v][y] is a set. We need to check if x is in lines[v_y][y].
    # This is equivalent to D[y][x] == v_y.
    # So forbidden check: exists y critical s.t. D[y][x] == v_y.
    
    U = np.full(n_points, (1 << 50) - 1, dtype=np.uint64)
    S_mask = np.zeros(n_points, dtype=bool)
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 500:
            break
            
        # Find criticals
        # U[y] is power of 2 and non-zero
        criticals = np.where((U > 0) & (U & (U - 1) == 0))[0]
        
        if len(criticals) == 0:
            # No criticals, any x with U[x] != 0 and not in S is valid
            # But we still need to check U[x] != 0
            pass
        else:
            # For each critical y, v_y is the direction.
            # x is forbidden if D[y][x] == v_y for any critical y.
            # This is equivalent to: x is in the union of lines[v_y][y].
            # We can compute a mask of forbidden points.
            forbidden_mask = np.zeros(n_points, dtype=bool)
            for y in criticals:
                v_y = int(np.log2(U[y])) # Slow?
                # Better: (U[y] & -U[y]).bit_length() - 1
                v_y = (U[y] & -U[y]).bit_length() - 1
                # D[y, :] == v_y
                forbidden_mask |= (D[y, :] == v_y)
        
        # Candidates: not in S, not forbidden, U[x] != 0
        valid_mask = (~S_mask) & (~forbidden_mask) & (U > 0)
        
        if not np.any(valid_mask):
            break
            
        # Pick best x
        # Heuristic: max popcount of U[x]
        # We can compute popcount for all valid candidates?
        # U is uint64.
        # We can just pick the first one? Or random?
        # Let's pick the one with max U[x] value? No.
        # Let's pick random to avoid bias.
        candidates = np.where(valid_mask)[0]
        best_x = candidates[np.random.choice(len(candidates))]
        
        S_mask[best_x] = True
        
        # Update U
        # U[y] &= ~(1 << D[y][best_x])
        masks = 1 << D[:, best_x]
        U &= ~masks
        # Don't update U[best_x]
        U[best_x] = (1 << 50) - 1 # Restore? No, U[best_x] was not changed by the mask?
        # D[best_x, best_x] is 0. masks[best_x] = 1 << 0 = 1.
        # U[best_x] &= ~1.
        # But U[best_x] should NOT be changed by adding best_x to S.
        # So we must restore U[best_x].
        # Wait, the update `U &= ~masks` applies to all y.
        # For y=best_x, D[best_x][best_x] is 0.
        # So bit 0 is cleared.
        # But adding best_x to S does NOT clear bit 0 of U[best_x].
        # So we must set U[best_x] back to its previous value.
        # Previous value is U[best_x] before update.
        # We can store it.
        
        # Actually, simpler:
        # U[best_x] is unchanged.
        # So:
        old_U_best = U[best_x]
        U &= ~(1 << D[:, best_x])
        U[best_x] = old_U_best

    # Construct result
    points = []
    for x in range(n_points):
        if not S_mask[x]:
            xa, xb = to_coords(x)
            points.append([[xa[0], xa[1]], [xb[0], xb[1]]])
            
    return np.array(points, dtype=np.int64)
