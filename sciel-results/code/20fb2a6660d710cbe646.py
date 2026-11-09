import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    S_total = np.sum(adj)
    
    def get_cut(x):
        xAx = x @ (adj @ x)
        return 0.25 * (S_total - xAx)
    
    # Spectral initialization
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    
    rng = np.random.default_rng(42)
    
    def local_search_1flip(x, max_iters=10000):
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            if max_gain <= 0:
                break
            v = x[i]
            x[i] = -v
            C += max_gain
            S -= 2 * v * adj[:, i]
            iters += 1
        return x, C, iters
    
    def local_search_2swap(x, max_iters=5000):
        S = adj @ x
        C = get_cut(x)
        iters = 0
        while iters < max_iters:
            gains_1 = x * S
            i = np.argmax(gains_1)
            if gains_1[i] <= 0:
                found = False
                for _ in range(20):
                    j = rng.integers(0, n)
                    if j == i:
                        continue
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * x[i] * x[j]
                    if gain > 0:
                        vi = x[i]
                        vj = x[j]
                        x[i] = -vi
                        x[j] = -vj
                        C += gain
                        S -= 2 * vi * adj[:, i]
                        S -= 2 * vj * adj[:, j]
                        gains_1 = x * S
                        iters += 1
                        found = True
                        break
                if not found:
                    break
            else:
                j_gains = gains_1.copy()
                interaction = 2 * x[i] * x * adj[i, :]
                j_gains[i] = -1e9
                j_gains -= interaction
                j = np.argmax(j_gains)
                if j == i or j_gains[j] <= 0:
                    v = x[i]
                    x[i] = -v
                    C += gains_1[i]
                    S -= 2 * v * adj[:, i]
                    gains_1 = x * S
                    iters += 1
                else:
                    vi = x[i]
                    vj = x[j]
                    x[i] = -vi
                    x[j] = -vj
                    gain = gains_1[i] + gains_1[j] - 2 * adj[i, j] * vi * vj
                    C += gain
                    S -= 2 * vi * adj[:, i]
                    S -= 2 * vj * adj[:, j]
                    gains_1 = x * S
                    iters += 1
        return x, C, iters
    
    def simulated_annealing(x, start_temp=5.0, cooling=0.95, min_temp=0.01, max_iters=5000):
        S = adj @ x
        C = get_cut(x)
        temp = start_temp
        iters = 0
        best_C = C
        best_x = x.copy()
        
        while temp > min_temp and iters < max_iters:
            if time.time() > deadline - 5:
                break
            
            i = rng.integers(0, n)
            v = x[i]
            
            gain = x[i] * S[i]
            
            if gain > 0 or (temp > 0 and rng.random() < np.exp(gain / temp)):
                x[i] = -v
                C += gain
                S -= 2 * v * adj[:, i]
                iters += 1
                
                if C > best_C:
                    best_C = C
                    best_x = x.copy()
            
            temp *= cooling
        
        return best_x, best_C, iters
    
    def admm_solve(rho=1.0, max_iters=200, tol=1e-4):
        """ADMM for MaxCut: minimize -0.25*(S_total - x^T A x) s.t. x in {+1,-1}^n
        Using the formulation: minimize -0.5 * x^T A x + (rho/2)||x - z||^2 + dual
        where z is constrained to {+1,-1}^n
        """
        # We want to maximize x^T A x / 2 subject to x_i in {+1,-1}
        # ADMM: min_x -0.5*x^T A x + (rho/2)||x - z + u||^2
        # z update: z = sign(x + u)
        # u update: u = u + x - z
        
        # Initialize x randomly
        x = rng.choice([-1.0, 1.0], size=n)
        z = x.copy()
        u = np.zeros(n)
        
        # Precompute for x-update: (rho*I - A) x = rho*z - rho*u - A@x_old... 
        # Actually the x-update solves: minimize -0.5*x^T A x + (rho/2)||x - (z - u)||^2
        # Taking derivative: -A x + rho(x - z + u) = 0
        # (rho*I - A) x = rho*(z - u)
        # So x = (rho*I - A)^{-1} * rho * (z - u)
        
        # For efficiency, we can solve (rho*I - A) x = rho*(z-u)
        # Note: rho*I - A may not be positive definite, but let's try
        
        try:
            # Precompute the factorization once if possible
            M = rho * np.eye(n) - adj.astype(np.float64)
            # Use Cholesky if possible, otherwise LU
            try:
                L_fac = np.linalg.cholesky(M)
                def solve_M(b):
                    y = np.linalg.solve(L_fac, b)
                    return np.linalg.solve(L_fac.T, y)
            except:
                # Use LU
                from scipy.linalg import lu_factor, lu_solve
                lu, piv = lu_factor(M)
                def solve_M(b):
                    return lu_solve((lu, piv), b)
        except:
            return None, None
        
        best_C = -np.inf
        best_x = None
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # x-update
            rhs = rho * (z - u)
            x = solve_M(rhs)
            
            # z-update
            z_new = np.sign(x + u)
            z_new[z_new == 0] = 1
            
            # Check convergence
            x_res = np.linalg.norm(x - z_new, 2)
            z_res = np.linalg.norm(z_new - z, 2)
            
            # u-update
            u = u + x - z_new
            z = z_new
            
            # Compute cut
            x_int = z.astype(np.int8)
            C = get_cut(x_int)
            
            if C > best_C:
                best_C = C
                best_x = x_int.copy()
            
            if x_res < tol and z_res < tol:
                break
        
        if best_x is not None:
            return best_x, best_C
        return None, None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Spectral init (Fiedler vector)
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        x_spec = np.sign(fiedler)
        x_spec[x_spec == 0] = 1
        inits.append(x_spec.astype(np.int8))
        
        # 2. Second Fiedler vector variant
        if len(vecs) > 2:
            fiedler2 = vecs[:, 2]
            x_spec2 = np.sign(fiedler2)
            x_spec2[x_spec2 == 0] = 1
            inits.append(x_spec2.astype(np.int8))
    except:
        x_spec = np.ones(n, dtype=np.int8)
        inits.append(x_spec.copy())
    
    # 3. Alternating init
    alt = np.ones(n, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    
    # 4. Random inits
    for _ in range(8):
        inits.append(rng.choice([-1, 1], size=n).astype(np.int8))
    
    # 5. Perturbed spectral inits
    for _ in range(4):
        x_pert = x_spec.copy()
        num_flips = max(1, n // 5)
        flip_idx = rng.choice(n, size=num_flips, replace=False)
        x_pert[flip_idx] *= -1
        inits.append(x_pert.astype(np.int8))
    
    best_C = -1
    best_x = None
    
    # Phase 1: Local search on all initializations
    for x_init in inits:
        if time.time() > deadline - 30:
            break
        
        x_curr = x_init.copy()
        x_ls, C_ls, _ = local_search_1flip(x_curr)
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    if time.time() > deadline - 20:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 2: 2-swap polish on best
    x_curr = best_x.copy()
    x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=3000)
    if C_ls2 > best_C:
        best_C = C_ls2
        best_x = x_ls2.copy()
    
    if time.time() > deadline - 15:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 3: SA polish on best to escape local optima
    x_sa, C_sa, _ = simulated_annealing(best_x.copy(), start_temp=5.0, cooling=0.95, min_temp=0.01, max_iters=3000)
    if C_sa > best_C:
        best_C = C_sa
        best_x = x_sa.copy()
    
    # Local search after SA
    x_ls3, C_ls3, _ = local_search_1flip(best_x.copy())
    if C_ls3 > best_C:
        best_C = C_ls3
        best_x = x_ls3.copy()
    
    if time.time() > deadline - 10:
        return ((best_x + 1) // 2).astype(int)
    
    # Phase 4: More diverse restarts with local search
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i % 4 == 0:
            x_curr = rng.choice([-1, 1], size=n).astype(np.int8)
        elif i % 4 == 1:
            x_curr = best_x.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        elif i % 4 == 2:
            x_curr = x_spec.copy()
            num_flips = max(1, n // 5)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        else:
            x_curr = best_x.copy()
            num_flips = max(1, n // 20)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        
        x_ls, C_ls, _ = local_search_1flip(x_curr)
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    # Phase 5: ADMM-based global optimization attempts
    if time.time() < deadline - 15:
        # Try ADMM with different rho values and initializations
        for rho in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline - 5:
                break
            
            # Try ADMM from spectral init
            x_admm, C_admm = admm_solve(rho=rho, max_iters=100, tol=1e-4)
            if x_admm is not None and C_admm > best_C:
                # Polish with local search
                x_ls, C_ls, _ = local_search_1flip(x_admm.copy())
                if C_ls > best_C:
                    best_C = C_ls
                    best_x = x_ls.copy()
            
            # Try ADMM from best init
            if time.time() < deadline - 5:
                x_admm, C_admm = admm_solve(rho=rho, max_iters=100, tol=1e-4)
                if x_admm is not None and C_admm > best_C:
                    x_ls, C_ls, _ = local_search_1flip(x_admm.copy())
                    if C_ls > best_C:
                        best_C = C_ls
                        best_x = x_ls.copy()
    
    # Phase 6: Final 2-swap + SA polish if time permits
    if time.time() < deadline - 5:
        x_curr = best_x.copy()
        x_ls2, C_ls2, _ = local_search_2swap(x_curr, max_iters=2000)
        if C_ls2 > best_C:
            best_C = C_ls2
            best_x = x_ls2.copy()
        
        if time.time() < deadline - 3:
            x_sa, C_sa, _ = simulated_annealing(best_x.copy(), start_temp=3.0, cooling=0.95, min_temp=0.01, max_iters=1500)
            if C_sa > best_C:
                best_C = C_sa
                best_x = x_sa.copy()
            
            # Final local search
            x_ls_final, C_ls_final, _ = local_search_1flip(best_x.copy())
            if C_ls_final > best_C:
                best_C = C_ls_final
                best_x = x_ls_final.copy()
    
    if best_x is None:
        best_x = rng.choice([-1, 1], size=n).astype(np.int8)
    
    labels = (best_x + 1) // 2
    return labels.astype(int)
