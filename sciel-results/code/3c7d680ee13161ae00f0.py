<<<<<<< SEARCH
def _topology_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with Delaunay topology constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    from scipy.spatial import Delaunay
    
    def _get_delaunay_edges(c):
        try:
            tri = Delaunay(c)
            edges = tri.simplices
            edge_set = set()
            for simplex in edges:
                i, j, k = simplex
                edge_set.add((min(i,j), max(i,j)))
                edge_set.add((min(i,k), max(i,k)))
                edge_set.add((min(j,k), max(j,k)))
            edge_list = np.array(list(edge_set), dtype=np.int32)
            if edge_list.size == 0:
                edge_list = np.empty((0, 2), dtype=np.int32)
            return edge_list
        except Exception:
            # Fallback: KNN
            tree = cKDTree(c)
            _, indices = tree.query(c, k=6)
            edge_set = set()
            for i in range(n):
                for j in indices[i]:
                    if i != j:
                        edge_set.add((min(i,j), max(i,j)))
            edge_list = np.array(list(edge_set), dtype=np.int32)
            if edge_list.size == 0:
                edge_list = np.empty((0, 2), dtype=np.int32)
            return edge_list

    # Initial topology
    edge_list = _get_delaunay_edges(best_c)

    def objective_and_constraints(params):
        c = params[:2*n].reshape(-1, 2)
        r = params[2*n]
        obj = -r
        
        bd = 0.5 - np.abs(c) - r
        bd_flat = bd.ravel()
        
        if len(edge_list) > 0:
            ci = c[edge_list[:, 0]]
            cj = c[edge_list[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pair_cons = dists - 2 * r
        else:
            pair_cons = np.array([])
        
        constraints = np.concatenate([bd_flat, pair_cons])
        return obj, constraints

    def jac(params):
        c = params[:2*n].reshape(-1, 2)
        r = params[2*n]
        n_cons = 2*n + len(edge_list)
        jac = np.zeros((n_cons, 2*n + 1))
        
        signs = -np.sign(c)
        for i in range(n):
            jac[2*i, 2*i] = signs[i, 0]
            jac[2*i+1, 2*i+1] = signs[i, 1]
        jac[:2*n, 2*n] = -1.0
        
        if len(edge_list) > 0:
            ci = c[edge_list[:, 0]]
            cj = c[edge_list[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            
            row_start = 2*n
            for idx in range(len(edge_list)):
                i, j = edge_list[idx]
                row = row_start + idx
                
                jac[row, 2*i] = unit_vecs[idx, 0]
                jac[row, 2*i+1] = unit_vecs[idx, 1]
                jac[row, 2*j] = -unit_vecs[idx, 0]
                jac[row, 2*j+1] = -unit_vecs[idx, 1]
                jac[row, 2*n] = -2.0
        
        return jac

    def obj_grad(x):
        g = np.zeros(2*n + 1)
        g[2*n] = -1.0
        return g

    def cons_func(x):
        return objective_and_constraints(x)[1]
    
    def cons_jac(x):
        return jac(x)

    x0 = np.concatenate([best_c.flatten(), [best_r]])
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2*n) + [(0.0, 0.5)]
    
    try:
        result = minimize(
            lambda x: objective_and_constraints(x)[0],
            x0,
            method='SLSQP',
            bounds=bounds,
            jac=obj_grad,
            constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
            options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
        )
        
        if result.success:
            c_new = result.x[:2*n].reshape(-1, 2)
            r_new = result.x[2*n]
            
            c_new = np.clip(c_new, -0.5 + 1e-12, 0.5 - 1e-12)
            r_verify = _compute_radius(c_new)
            
            if r_verify > best_r:
                best_c = c_new
                best_r = r_verify
                
                # Update topology and try again if time permits
                if time.time() < start_time + max_time - 10:
                    edge_list = _get_delaunay_edges(best_c)
                    x0 = np.concatenate([best_c.flatten(), [best_r]])
                    result2 = minimize(
                        lambda x: objective_and_constraints(x)[0],
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        jac=obj_grad,
                        constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
                        options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
                    )
                    if result2.success:
                        c_new2 = result2.x[:2*n].reshape(-1, 2)
                        r_new2 = result2.x[2*n]
                        c_new2 = np.clip(c_new2, -0.5 + 1e-12, 0.5 - 1e-12)
                        r_verify2 = _compute_radius(c_new2)
                        if r_verify2 > best_r:
                            best_c = c_new2
                            best_r = r_verify2

    except Exception:
        pass

    return best_c, best_r
=======
def _topology_polish(c0, r0, max_time=30.0):
    """
    High-precision polish using SLSQP with active constraint detection.
    Uses only the most binding constraints to keep the problem small.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N
    
    def _get_active_constraints(c, r, margin=0.002):
        """Get active boundary and pair constraints."""
        # Active boundary constraints
        bd = 0.5 - np.abs(c)
        active_bd = np.where(bd.ravel() < r + margin)[0]
        
        # Active pair constraints using KDTree
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + margin, output_type='ndarray')
        
        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            diffs = c[i_idx] - c[j_idx]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            active_mask = dists < 2 * r + margin
            active_pairs = pairs[active_mask]
        else:
            active_pairs = np.empty((0, 2), dtype=np.int32)
        
        return active_bd, active_pairs
    
    def _run_slsqp(c_init, r_init, max_iter=300):
        """Run SLSQP with active constraints."""
        active_bd, active_pairs = _get_active_constraints(c_init, r_init)
        
        n_bd = len(active_bd)
        n_pairs = len(active_pairs)
        n_cons = n_bd + n_pairs
        
        # Map boundary indices to circle/coord
        bd_circles = active_bd // 2
        bd_coords = active_bd % 2
        
        def obj(x):
            return -x[2*n]
        
        def obj_grad(x):
            g = np.zeros(2*n + 1)
            g[2*n] = -1.0
            return g
        
        def cons_func(x):
            c = x[:2*n].reshape(-1, 2)
            r = x[2*n]
            cons = []
            
            # Boundary constraints
            if n_bd > 0:
                bd_vals = 0.5 - np.abs(c)
                bd_flat = bd_vals.ravel()
                cons.append(bd_flat[active_bd] - r)
            
            # Pair constraints
            if n_pairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                cons.append(dists - 2 * r)
            
            return np.concatenate(cons) if cons else np.array([0.0])
        
        def cons_jac(x):
            c = x[:2*n].reshape(-1, 2)
            r = x[2*n]
            J = np.zeros((n_cons, 2*n + 1))
            row = 0
            
            # Boundary Jacobian
            if n_bd > 0:
                for k in range(n_bd):
                    ci = bd_circles[k]
                    coord = bd_coords[k]
                    sign = -np.sign(c[ci, coord])
                    if sign == 0:
                        sign = 1.0
                    J[row, 2*ci + coord] = sign
                    J[row, 2*n] = -1.0
                    row += 1
            
            # Pair Jacobian
            if n_pairs > 0:
                ci = c[active_pairs[:, 0]]
                cj = c[active_pairs[:, 1]]
                diffs = ci - cj
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                dists_safe = np.where(dists < 1e-12, 1e-12, dists)
                unit_vecs = diffs / dists_safe[:, np.newaxis]
                
                for k in range(n_pairs):
                    i, j = active_pairs[k]
                    J[row, 2*i] = unit_vecs[k, 0]
                    J[row, 2*i+1] = unit_vecs[k, 1]
                    J[row, 2*j] = -unit_vecs[k, 0]
                    J[row, 2*j+1] = -unit_vecs[k, 1]
                    J[row, 2*n] = -2.0
                    row += 1
            
            return J
        
        x0 = np.concatenate([c_init.flatten(), [r_init]])
        bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2*n) + [(0.0, 0.5)]
        
        try:
            result = minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                jac=obj_grad,
                constraints=[{'type': 'ineq', 'fun': cons_func, 'jac': cons_jac}],
                options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False}
            )
            
            if result.success:
                c_new = result.x[:2*n].reshape(-1, 2)
                r_new = result.x[2*n]
                c_new = np.clip(c_new, -0.5 + 1e-12, 0.5 - 1e-12)
                return c_new, r_new
            else:
                return c_init, r_init
        except Exception:
            return c_init, r_init
    
    # Multiple polish rounds
    for round_num in range(4):
        if time.time() > start_time + max_time:
            break
        
        c_new, r_new = _run_slsqp(best_c, best_r, max_iter=200)
        
        c_new = np.clip(c_new, -0.5 + 1e-12, 0.5 - 1e-12)
        r_verify = _compute_radius(c_new)
        
        if r_verify > best_r:
            best_c = c_new
            best_r = r_verify
    
    return best_c, best_r
>>>>>>> REPLACE

