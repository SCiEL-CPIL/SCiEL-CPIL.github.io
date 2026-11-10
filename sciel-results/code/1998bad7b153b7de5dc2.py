<<<<<<< SEARCH
    # Precompute: for each point and direction, the list of other 48 points on that line
    # affected[p_idx][d_idx] = array of 48 point indices
    affected = np.zeros((Q2, num_dirs, 48), dtype=np.int32)
    for p_idx in range(Q2):
        xa, xb, ya, yb = get_coords(p_idx)
        px = (xa, xb)
        py = (ya, yb)
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
                    n_idx = get_idx(nx[0], nx[1], ny[0], ny[1])
                    affected[p_idx, d_idx, count] = n_idx
                    count += 1
=======
    # Precompute: for each point and direction, the list of other 48 points on that line
    # affected[p_idx][d_idx] = array of 48 point indices
    # Vectorize using field structure: F_49 elements are (a, b) representing a + b*alpha
    # Point index: idx = xa*343 + xb*49 + ya*7 + yb
    # For direction v = (vx, vy), point x + t*v where t = (ta, tb)
    # (x + t*v)_a = xa + ta*vx_a + tb*vx_b*3 (mod 7)
    # (x + t*v)_b = xb + ta*vx_b + tb*vx_a (mod 7)
    # Similarly for y
    
    affected = np.zeros((Q2, num_dirs, 48), dtype=np.int32)
    
    # Precompute all t values (48 non-zero elements of F_49)
    t_vals = []
    for ta in range(p):
        for tb in range(p):
            if ta != 0 or tb != 0:
                t_vals.append((ta, tb))
    
    # Precompute for each direction and each t, the offset
    # offset[d_idx][t_idx] = (dx_a, dx_b, dy_a, dy_b)
    offsets = np.zeros((num_dirs, 48, 4), dtype=np.int8)
    for d_idx in range(num_dirs):
        vx, vy = directions[d_idx]
        for t_idx, (ta, tb) in enumerate(t_vals):
            tvx = mul((ta, tb), vx)
            tvy = mul((ta, tb), vy)
            offsets[d_idx, t_idx, 0] = tvx[0]
            offsets[d_idx, t_idx, 1] = tvx[1]
            offsets[d_idx, t_idx, 2] = tvy[0]
            offsets[d_idx, t_idx, 3] = tvy[1]
    
    # Vectorize: for each point, compute all affected indices
    for p_idx in range(Q2):
        xa, xb, ya, yb = get_coords(p_idx)
        for d_idx in range(num_dirs):
            # Add offsets to point coordinates
            dx_a = (xa + offsets[d_idx, :, 0]) % p
            dx_b = (xb + offsets[d_idx, :, 1]) % p
            dy_a = (ya + offsets[d_idx, :, 2]) % p
            dy_b = (yb + offsets[d_idx, :, 3]) % p
            n_idxs = dx_a * 343 + dx_b * 49 + dy_a * 7 + dy_b
            affected[p_idx, d_idx, :] = n_idxs
>>>>>>> REPLACE

