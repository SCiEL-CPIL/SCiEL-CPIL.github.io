import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute symmetric versions if needed (QAPLIB instances should be symmetric)
    a_sym = (a + a.T) / 2
    b_sym = (b + b.T) / 2

    def compute_cost(p):
        """Compute QAP cost for permutation p."""
        return int(np.sum(a_sym * b_sym[np.ix_(p, p)]))

    def spectral_init():
        """Initialize using Procrustes/MDS approach."""
        # Center the matrices
        a_centered = a_sym - a_sym.mean()
        b_centered = b_sym - b_sym.mean()
        
        # SVD for MDS
        try:
            U_a, S_a, Vt_a = np.linalg.svd(a_centered, full_matrices=False)
            U_b, S_b, Vt_b = np.linalg.svd(b_centered, full_matrices=False)
            
            # Use top k components
            k = min(20, n)
            coords_a = U_a[:, :k] * np.sqrt(S_a[:k])
            coords_b = U_b[:, :k] * np.sqrt(S_b[:k])
            
            # Procrustes: find rotation that best aligns coords_a to coords_b
            H = coords_a.T @ coords_b
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Apply rotation
            coords_a_rotated = coords_a @ R
            
            # Match by nearest neighbor or by sorting
            # Use a simple greedy matching based on rotated coordinates
            # Actually, let's use linear_sum_assignment on the distance matrix between coords
            dist_mat = np.zeros((n, n))
            for i in range(n):
                dist_mat[i] = np.sum((coords_b - coords_a_rotated[i]) ** 2, axis=1)
            
            _, p = linear_sum_assignment(dist_mat)
            return p
        except:
            # Fallback to simple heuristic
            a_sums = a_sym.sum(axis=1)
            b_sums = b_sym.sum(axis=1)
            heavy = np.argsort(-a_sums, kind="stable")
            central = np.argsort(b_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    def local_search(p, max_time=None, max_no_improve=2000):
        """
        Fast local search using O(n) delta computation for swaps.
        """
        current_p = p.copy()
        current_cost = compute_cost(current_p)
        
        # Precompute for delta calculation
        # delta for swapping positions i, j:
        # d[k] = b_sym[current_p[k], y] - b_sym[current_p[k], x] where x=cp[i], y=cp[j]
        # delta = 2 * sum_{k!=i,j} d[k] * (a_sym[i,k] - a_sym[j,k])
        #        + a_sym[i,j]*(b_sym[y,x]-b_sym[x,y]) + ... but b_sym is symmetric so first term is 0
        # Actually with symmetric a and b, the formula simplifies to:
        # delta = 2 * sum_{k!=i,j} (b_sym[cp[k], y] - b_sym[cp[k], x]) * (a_sym[i,k] - a_sym[j,k])
        
        no_improve = 0
        best_cost = current_cost
        best_p = current_p.copy()
        
        # Precompute a_diff for faster access? No, just use a_sym directly.
        
        while True:
            if max_time is not None and time.time() > max_time:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick random pair to swap
            i, j = np.random.choice(n, 2, replace=False)
            x = current_p[i]
            y = current_p[j]
            
            if x == y:
                continue
            
            # Compute delta
            # d[k] = b_sym[current_p[k], y] - b_sym[current_p[k], x]
            d = b_sym[current_p, y] - b_sym[current_p, x]
            coeff = a_sym[i, :] - a_sym[j, :]
            
            # Exclude k=i and k=j
            # d[i] and d[j] should not be included
            # coeff[i] and coeff[j] should not be included
            # But note: when k=i, the term involves a_sym[i,i] which is 0 in QAP usually
            # And b_sym[cp[i], y] - b_sym[cp[i], x] = b_sym[x,y] - b_sym[x,x]
            # This is not necessarily 0, so we must exclude them.
            
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            delta = 2.0 * np.sum(d[mask] * coeff[mask])
            
            # Also need to account for the direct interaction between i and j
            # Old: a_sym[i,j] * b_sym[x,y] + a_sym[j,i] * b_sym[y,x] = 2 * a_sym[i,j] * b_sym[x,y]
            # New: a_sym[i,j] * b_sym[y,x] + a_sym[j,i] * b_sym[x,y] = 2 * a_sym[i,j] * b_sym[y,x]
            # Since b_sym is symmetric, b_sym[x,y] = b_sym[y,x], so this term is unchanged.
            # So the delta formula above is complete for symmetric a and b.
            
            if delta < -1e-9:  # Small tolerance
                # Accept the swap
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += int(delta)
                no_improve = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            else:
                no_improve += 1
        
        return best_p, best_cost

    def simulated_annealing(p, max_time=None, T_initial=1000.0, cooling=0.9995, max_no_improve=5000):
        """
        Simulated Annealing with O(n) delta computation.
        """
        current_p = p.copy()
        current_cost = compute_cost(current_p)
        best_cost = current_cost
        best_p = current_p.copy()
        
        T = T_initial
        no_improve = 0
        
        while True:
            if max_time is not None and time.time() > max_time:
                break
            if T < 1e-4:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick random pair to swap
            i, j = np.random.choice(n, 2, replace=False)
            x = current_p[i]
            y = current_p[j]
            
            if x == y:
                T *= cooling
                continue
            
            # Compute delta
            d = b_sym[current_p, y] - b_sym[current_p, x]
            coeff = a_sym[i, :] - a_sym[j, :]
            
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            delta = 2.0 * np.sum(d[mask] * coeff[mask])
            
            if delta < 0:
                # Accept
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += int(delta)
                no_improve = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
            else:
                # Metropolis criterion
                if np.random.random() < np.exp(-delta / T):
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                    current_cost += int(delta)
                else:
                    no_improve += 1
            
            T *= cooling
        
        return best_p, best_cost

    # Step 1: Initialize using spectral method
    p_init = spectral_init()
    
    # Step 2: Local search from initial solution
    best_p, best_cost = local_search(p_init, max_time=start_time + 20.0)
    
    # Step 3: Multi-start with simulated annealing and local search
    rng = np.random.RandomState(42)
    
    # Allocate time: 50% for SA, 30% for LS, 20% for final polish
    sa_end = start_time + 70.0
    ls_end = start_time + 115.0
    
    # Phase 1: Simulated Annealing with periodic local search
    while time.time() < sa_end:
        # Create perturbed version of best solution
        p_perturb = best_p.copy()
        
        # Swap random number of pairs
        n_swaps = rng.randint(3, 10)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_perturb[i], p_perturb[j] = p_perturb[j], p_perturb[i]
        
        # Run SA with limited time
        time_per_sa = min(5.0, sa_end - time.time())
        if time_per_sa < 1.0:
            break
            
        p_sa, cost_sa = simulated_annealing(p_perturb, max_time=time.time() + time_per_sa, 
                                            T_initial=500.0, cooling=0.999)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa
    
    # Phase 2: Local search from best solution
    while time.time() < ls_end:
        # Create perturbed version
        p_perturb = best_p.copy()
        
        n_swaps = rng.randint(2, 8)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p_perturb[i], p_perturb[j] = p_perturb[j], p_perturb[i]
        
        time_per_ls = min(5.0, ls_end - time.time())
        if time_per_ls < 1.0:
            break
            
        p_ls, cost_ls = local_search(p_perturb, max_time=time.time() + time_per_ls,
                                     max_no_improve=1000)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Phase 3: Final polish with intensive local search
    p_final, cost_final = local_search(best_p, max_time=deadline, max_no_improve=10000)
    
    if cost_final < best_cost:
        best_cost = cost_final
        best_p = p_final
    
    return best_p.tolist()
