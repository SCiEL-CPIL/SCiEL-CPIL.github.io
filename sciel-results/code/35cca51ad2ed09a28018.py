<<<<<<< SEARCH
def _jiggle_polish(cx, cy, r, W, H, time_budget):
    """
    Jiggle positions to allow radius expansion.
    Move circles that are constrained, then re-expand radii.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        # Find circles that are close to their constraint
        min_margin = np.full(N, np.inf)
        for i in range(N):
            margin_wall = min(cx[i] - r[i], W - cx[i] - r[i], cy[i] - r[i], H - cy[i] - r[i])
            margin_neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                margin_neigh = min(margin_neigh, d - r[i] - r[j])
            min_margin[i] = min(margin_wall, margin_neigh)
        
        # Try to move circles with small margins
        moved = False
        for i in range(N):
            if min_margin[i] > 0.02:
                continue
            
            # Determine direction to move
            ml = cx[i] - r[i]
            mr = W - cx[i] - r[i]
            mb = cy[i] - r[i]
            mt = H - cy[i] - r[i]
            
            # Find nearest neighbor
            best_j = -1
            best_d = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < best_d:
                    best_d = d
                    best_j = j
            
            mg = best_d - r[i] - r[best_j] if best_j >= 0 else np.inf
            
            if mg <= ml and mg <= mr and mg <= mb and mg <= mt and best_j >= 0:
                dx = cx[i] - cx[best_j]
                dy = cy[i] - cy[best_j]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-8:
                    move_x = dx / d
                    move_y = dy / d
                else:
                    move_x = 1.0
                    move_y = 0.0
            elif ml <= mr and ml <= mb and ml <= mt:
                move_x = 1.0
                move_y = 0.0
            elif mr <= ml and mr <= mb and mr <= mt:
                move_x = -1.0
                move_y = 0.0
            elif mb <= ml and mb <= mr and mb <= mt:
                move_x = 0.0
                move_y = 1.0
            else:
                move_x = 0.0
                move_y = -1.0
            
            # Try small moves
            for step in [0.01, 0.005, 0.002, 0.001]:
                new_cx = cx[i] + move_x * step
                new_cy = cy[i] + move_y * step
                if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                    continue
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = new_cx - cx[j]
                    dy = new_cy - cy[j]
                    if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                        valid = False
                        break
                if valid:
                    cx[i] = new_cx
                    cy[i] = new_cy
                    moved = True
                    break
        
        if not moved:
            # Try random jiggles
            rng = np.random.default_rng(rnd)
            for i in range(N):
                if min_margin[i] > 0.01:
                    continue
                angle = rng.uniform(0, 2 * np.pi)
                for step in [0.01, 0.005, 0.002]:
                    new_cx = cx[i] + np.cos(angle) * step
                    new_cy = cy[i] + np.sin(angle) * step
                    if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                        continue
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = new_cx - cx[j]
                        dy = new_cy - cy[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if valid:
                        cx[i] = new_cx
                        cy[i] = new_cy
                        moved = True
                        break
                if moved:
                    break
        
        if not moved:
            break
        
        # Re-expand radii
        r_new = _expand_radii(cx, cy, W, H, time_budget)
        if np.sum(r_new) > np.sum(r) + 1e-9:
            r = r_new
    
    return cx, cy, r
=======
def _jiggle_polish(cx, cy, r, W, H, time_budget):
    """
    Jiggle positions to allow radius expansion.
    Move circles that are constrained, then re-expand radii.
    Uses vectorized distance computations for speed.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        # Vectorized margin computation
        # Wall margins
        ml = cx - r
        mr = W - cx - r
        mb = cy - r
        mt = H - cy - r
        margin_wall = np.minimum(np.minimum(ml, mr), np.minimum(mb, mt))
        
        # Neighbor margins (vectorized)
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        neighbor_margin = d - r_sum  # shape (N, N)
        min_margin_neigh = np.min(neighbor_margin, axis=1)
        
        min_margin = np.minimum(margin_wall, min_margin_neigh)
        
        # Try to move circles with small margins
        moved = False
        for i in range(N):
            if min_margin[i] > 0.02:
                continue
            
            # Determine direction to move
            ml_i = cx[i] - r[i]
            mr_i = W - cx[i] - r[i]
            mb_i = cy[i] - r[i]
            mt_i = H - cy[i] - r[i]
            
            # Find nearest neighbor
            dists_i = np.sqrt((cx - cx[i])**2 + (cy - cy[i])**2)
            dists_i[i] = np.inf
            best_j = int(np.argmin(dists_i))
            best_d = dists_i[best_j]
            mg = best_d - r[i] - r[best_j]
            
            if mg <= ml_i and mg <= mr_i and mg <= mb_i and mg <= mt_i:
                dx = cx[i] - cx[best_j]
                dy = cy[i] - cy[best_j]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-8:
                    move_x = dx / d
                    move_y = dy / d
                else:
                    move_x = 1.0
                    move_y = 0.0
            elif ml_i <= mr_i and ml_i <= mb_i and ml_i <= mt_i:
                move_x = 1.0
                move_y = 0.0
            elif mr_i <= ml_i and mr_i <= mb_i and mr_i <= mt_i:
                move_x = -1.0
                move_y = 0.0
            elif mb_i <= ml_i and mb_i <= mr_i and mb_i <= mt_i:
                move_x = 0.0
                move_y = 1.0
            else:
                move_x = 0.0
                move_y = -1.0
            
            # Try small moves
            for step in [0.01, 0.005, 0.002, 0.001]:
                new_cx = cx[i] + move_x * step
                new_cy = cy[i] + move_y * step
                if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                    continue
                # Vectorized validity check
                dx_all = new_cx - cx
                dy_all = new_cy - cy
                dx_all[i] = np.inf
                dy_all[i] = np.inf
                d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
                valid = np.all(d_all >= (r[i] + r) - 1e-9)
                if valid:
                    cx[i] = new_cx
                    cy[i] = new_cy
                    moved = True
                    break
        
        if not moved:
            # Try random jiggles
            rng = np.random.default_rng(rnd)
            for i in range(N):
                if min_margin[i] > 0.01:
                    continue
                angle = rng.uniform(0, 2 * np.pi)
                for step in [0.01, 0.005, 0.002]:
                    new_cx = cx[i] + np.cos(angle) * step
                    new_cy = cy[i] + np.sin(angle) * step
                    if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                        continue
                    # Vectorized validity check
                    dx_all = new_cx - cx
                    dy_all = new_cy - cy
                    dx_all[i] = np.inf
                    dy_all[i] = np.inf
                    d_all = np.sqrt(dx_all * dx_all + dy_all * dy_all)
                    valid = np.all(d_all >= (r[i] + r) - 1e-9)
                    if valid:
                        cx[i] = new_cx
                        cy[i] = new_cy
                        moved = True
                        break
                if moved:
                    break
        
        if not moved:
            break
        
        # Re-expand radii
        r_new = _expand_radii(cx, cy, W, H, time_budget)
        if np.sum(r_new) > np.sum(r) + 1e-9:
            r = r_new
    
    return cx, cy, r


def _high_precision_polish(cx, cy, r, W, H, time_budget):
    """
    High-precision polish: use the smooth objective with tighter tolerances
    and more iterations to squeeze out additional radius sum.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    best_sum = np.sum(r)
    
    # Build initial z
    z = np.zeros(3 * N + 1)
    z[0::3][:N] = cx
    z[1::3][:N] = cy
    z[2::3][:N] = r
    z[-1] = W
    
    # Try multiple beta values for the smooth penalty
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > time_budget:
            break
        
        z_opt = _slsqp_smooth_refine(z, time_budget, maxiter=300, beta=beta)
        
        x_opt = z_opt[0::3][:N]
        y_opt = z_opt[1::3][:N]
        r_opt = z_opt[2::3][:N]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt
        
        if w_opt < 0.2 or h_opt < 0.2:
            continue
        
        # Expand radii
        r_expanded = _expand_radii(x_opt, y_opt, w_opt, h_opt, time_budget)
        
        if _is_valid(x_opt, y_opt, r_expanded, w_opt, h_opt):
            val = float(np.sum(r_expanded))
            if val > best_sum + 1e-9:
                best_sum = val
                cx = x_opt.copy()
                cy = y_opt.copy()
                r = r_expanded.copy()
                # Update z for next iteration
                z[0::3][:N] = cx
                z[1::3][:N] = cy
                z[2::3][:N] = r
                z[-1] = w_opt
    
    # Final tight SLSQP with standard objective
    if time.time() < time_budget - 2:
        z_opt = _slsqp_refine(z, time_budget, maxiter=200)
        x_opt = z_opt[0::3][:N]
        y_opt = z_opt[1::3][:N]
        r_opt = z_opt[2::3][:N]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt
        
        if w_opt >= 0.2 and h_opt >= 0.2:
            r_expanded = _expand_radii(x_opt, y_opt, w_opt, h_opt, time_budget)
            if _is_valid(x_opt, y_opt, r_expanded, w_opt, h_opt):
                val = float(np.sum(r_expanded))
                if val > best_sum + 1e-9:
                    best_sum = val
                    cx = x_opt.copy()
                    cy = y_opt.copy()
                    r = r_expanded.copy()
    
    return cx, cy, r
>>>>>>> REPLACE

