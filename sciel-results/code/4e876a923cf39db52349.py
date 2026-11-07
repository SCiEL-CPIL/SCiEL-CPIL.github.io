<<<<<<< SEARCH
    # Grid of candidate depths
    # Use a fine grid; the optimal depth is typically small
    n_grid = 200
    max_depth = 8.0
    depth_grid = np.linspace(1e-4, max_depth, n_grid)
=======
    # Grid of candidate depths
    # Use a fine grid; the optimal depth is typically small
    n_grid = 300
    max_depth = 10.0
    depth_grid = np.linspace(1e-4, max_depth, n_grid)
>>>>>>> REPLACE
<<<<<<< SEARCH
            # Vectorize: (n_grid^2, n_interior)
            PB = p_fill[:, None] * np.ones((1, n_grid))  # (n_grid, n_grid)
            PA = np.ones((n_grid, 1)) * p_fill[None, :]  # (n_grid, n_grid)
            DB = depth_grid[:, None] * np.ones((1, n_grid))  # (n_grid, n_grid)
            DA = np.ones((n_grid, 1)) * depth_grid[None, :]  # (n_grid, n_grid)
            
            PB_flat = PB.ravel()  # (n_grid^2,)
            PA_flat = PA.ravel()  # (n_grid^2,)
            DB_flat = DB.ravel()  # (n_grid^2,)
            DA_flat = DA.ravel()  # (n_grid^2,)
            
            # (n_grid^2, n_interior)
            vals = (Vq[None, :] + 
                    PB_flat[:, None] * (DB_flat[:, None] + delta1[None, :]) + 
                    PA_flat[:, None] * (DA_flat[:, None] - delta2[None, :]) - 
                    PB_flat[:, None] * PA_flat[:, None] * delta12[None, :])
            
            best_idx_flat = np.argmax(vals, axis=0)
            best_val = vals[best_idx_flat, np.arange(n_interior)]
            
            # Map back to grid
            best_i = best_idx_flat // n_grid
            best_j = best_idx_flat % n_grid
=======
            # Vectorize: (n_grid^2, n_interior)
            PB_flat = np.tile(p_fill, n_grid)  # (n_grid^2,)
            PA_flat = np.repeat(p_fill, n_grid)  # (n_grid^2,)
            DB_flat = np.tile(depth_grid, n_grid)  # (n_grid^2,)
            DA_flat = np.repeat(depth_grid, n_grid)  # (n_grid^2,)
            
            # (n_grid^2, n_interior)
            vals = (Vq[None, :] + 
                    PB_flat[:, None] * (DB_flat[:, None] + delta1[None, :]) + 
                    PA_flat[:, None] * (DA_flat[:, None] - delta2[None, :]) - 
                    PB_flat[:, None] * PA_flat[:, None] * delta12[None, :])
            
            best_idx_flat = np.argmax(vals, axis=0)
            best_val = vals[best_idx_flat, np.arange(n_interior)]
            
            # Map back to grid
            best_i = best_idx_flat // n_grid
            best_j = best_idx_flat % n_grid
>>>>>>> REPLACE
<<<<<<< SEARCH
    return bid_table, ask_table


def _get_policy_table(params):
=======
    # Final polish: refine depths around the grid solution
    # For each step and inventory, do a local 1D optimization
    for n in range(steps):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q_idx == 0:
                # Only bid
                d_bid = bid_table[n, q_idx]
                if np.isinf(d_bid):
                    continue
                # Local refinement around d_bid
                local_grid = np.linspace(max(1e-4, d_bid - 0.1), min(max_depth, d_bid + 0.1), 50)
                p_local = 1.0 - np.exp(-A * np.exp(-k * local_grid) * dt)
                q_next_idx = q_idx + 1
                diff = V[q_next_idx] - V[q_idx]
                vals = diff * p_local - local_grid * p_local
                best_local = np.argmax(vals)
                bid_table[n, q_idx] = local_grid[best_local]
            elif q_idx == n_inv - 1:
                # Only ask
                d_ask = ask_table[n, q_idx]
                if np.isinf(d_ask):
                    continue
                local_grid = np.linspace(max(1e-4, d_ask - 0.1), min(max_depth, d_ask + 0.1), 50)
                p_local = 1.0 - np.exp(-A * np.exp(-k * local_grid) * dt)
                q_next_idx = q_idx - 1
                diff = V[q_next_idx] - V[q_idx]
                vals = diff * p_local - local_grid * p_local
                best_local = np.argmax(vals)
                ask_table[n, q_idx] = local_grid[best_local]
            else:
                # Both bid and ask
                d_bid = bid_table[n, q_idx]
                d_ask = ask_table[n, q_idx]
                if np.isinf(d_bid) or np.isinf(d_ask):
                    continue
                
                # 2D local refinement
                local_b = np.linspace(max(1e-4, d_bid - 0.1), min(max_depth, d_bid + 0.1), 20)
                local_a = np.linspace(max(1e-4, d_ask - 0.1), min(max_depth, d_ask + 0.1), 20)
                
                LB, LA = np.meshgrid(local_b, local_a, indexing='ij')
                PB = 1.0 - np.exp(-A * np.exp(-k * LB) * dt)
                PA = 1.0 - np.exp(-A * np.exp(-k * LA) * dt)
                
                Vq = V[q_idx]
                Vq1 = V[q_idx + 1]
                Vqm1 = V[q_idx - 1]
                delta1 = Vq1 - Vq
                delta2 = Vq - Vqm1
                delta12 = Vq1 + Vqm1 - 2 * Vq
                
                vals = (Vq + 
                        PB * (LB + delta1) + 
                        PA * (LA - delta2) - 
                        PB * PA * delta12)
                
                best_idx = np.unravel_index(np.argmax(vals), vals.shape)
                bid_table[n, q_idx] = local_b[best_idx[0]]
                ask_table[n, q_idx] = local_a[best_idx[1]]
    
    return bid_table, ask_table


def _get_policy_table(params):
>>>>>>> REPLACE
