"""QAP solver for tho150 using local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Precompute row sums for initial heuristics
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Convert to float for faster arithmetic in delta computation
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost of permutation p."""
        M = b[np.ix_(p, p)]
        return int(np.sum(a * M))
    
    def local_search(p, deadline):
        """Steepest descent local search with efficient delta computation."""
        cost = compute_cost(p)
        improved = True
        M = b_f[np.ix_(p, p)]  # M[j,k] = b[p[j], p[k]]
        
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                if time.time() > deadline:
                    break
                
                pi = p[i]
                row_ai = a_f[i, :]
                col_ai = a_f[:, i]
                b_pi_p = b_f[pi, p]
                b_p_pi = b_f[p, pi]
                
                # Compute deltas for all j in vectorized fashion
                # delta(j) = (row_ai - row_aj) @ (M[j,:] - b_pi_p) + (col_ai - col_aj) @ (M[:,j] - b_p_pi)
                #          + (a[i,i] - a[j,j]) * (b[pj,pj] - b[pi,pi])
                
                # Precompute M[j,:] - b_pi_p for all j: M - b_pi_p[None, :]
                M_minus_bpi = M - b_pi_p[None, :]
                # Precompute M[:,j] - b_p_pi for all j: M.T - b_p_pi[:, None]
                MT_minus_bpiT = M.T - b_p_pi[:, None]
                
                # For each j, we need row_aj and col_aj
                # deltas[j] = row_ai @ M_minus_bpi[j,:] - row_aj @ M_minus_bpi[j,:]
                #           + col_ai @ MT_minus_bpiT[:,j] - col_aj @ MT_minus_bpiT[:,j]
                #           + (a[i,i] - a[j,j]) * (b_f[p[j],p[j]] - b_f[pi,pi])
                
                # Vectorize: row_ai @ M_minus_bpi[j,:] for all j = (M_minus_bpi.T @ row_ai)[j]
                term1_i = M_minus_bpi.T @ row_ai  # term1_i[j] = row_ai @ M_minus_bpi[j,:]
                term2_i = MT_minus_bpiT @ col_ai  # term2_i[j] = col_ai @ MT_minus_bpiT[:,j]
                
                # For the j-dependent terms, we need row_aj @ M_minus_bpi[j,:] and col_aj @ MT_minus_bpiT[:,j]
                # This is harder to vectorize. Let's compute it in a loop but efficiently.
                
                deltas = np.zeros(n, dtype=np.float64)
                b_diag_diff = b_f[p, p] - b_f[pi, pi]  # b[pj,pj] - b[pi,pi] for all j
                
                for j in range(n):
                    if j == i:
                        continue
                    row_aj = a_f[j, :]
                    col_aj = a_f[:, j]
                    
                    d_i = term1_i[j] + term2_i[j]
                    d_j = -(row_aj @ M_minus_bpi[j, :]) - (col_aj @ MT_minus_bpiT[:, j])
                    diag = (a_f[i, i] - a_f[j, j]) * b_diag_diff[j]
                    
                    deltas[j] = d_i + d_j + diag
                
                # Find best j > i
                if i + 1 < n:
                    sub = deltas[i+1:]
                    best_idx = np.argmin(sub)
                    best_val = sub[best_idx]
                    if best_val < best_delta:
                        best_delta = best_val
                        best_i = i
                        best_j = i + 1 + best_idx
            
            if best_i >= 0:
                # Perform swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += int(best_delta)
                improved = True
                # Update M
                M = b_f[np.ix_(p, p)]
        
        return p, cost
    
    def spectral_initialization(a, b, n, rng):
        """Generate initial solution using spectral methods."""
        # Compute eigenvectors of flow and distance matrices
        try:
            # For flow matrix, use top eigenvectors
            flow_eigvals, flow_eigvecs = np.linalg.eigh(a_f)
            dist_eigvals, dist_eigvecs = np.linalg.eigh(b_f)
            
            # Take top k eigenvectors
            k = min(20, n)
            flow_top = flow_eigvecs[:, -k:]
            dist_top = dist_eigvecs[:, -k:]
            
            # Create similarity measures
            flow_sim = flow_top @ flow_top.T
            dist_sim = dist_top @ dist_top.T
            
            # Sort facilities by flow centrality (sum of absolute flow eigenvector values)
            flow_centrality = np.abs(flow_top).sum(axis=1)
            dist_centrality = np.abs(dist_top).sum(axis=1)
            
            fac_order = np.argsort(-flow_centrality, kind='stable')
            loc_order = np.argsort(dist_centrality, kind='stable')
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            
            # Add some randomness
            if rng.random() < 0.3:
                # Randomly swap a few elements
                for _ in range(rng.integers(3, 10)):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                        
        except Exception:
            # Fallback to rank matching
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            
        return p
    
    def make_initial_solution(method, rng):
        """Create an initial permutation using different strategies."""
        if method == 0:
            # Rank matching: heavy flow -> central location
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 1:
            # Reverse rank
            heavy = np.argsort(a_row_sums, kind='stable')
            central = np.argsort(-b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 2:
            # Random
            p = rng.permutation(n).astype(np.int64)
        elif method == 3:
            # Sort by a_row_sums / b_row_sums ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b_row_sums > 0, a_row_sums / np.maximum(b_row_sums, 1), 0)
            fac_order = np.argsort(-ratio, kind='stable')
            loc_order = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
        elif method == 4:
            # Spectral initialization
            p = spectral_initialization(a, b, n, rng)
        elif method == 5:
            # Spectral with different eigenvector count
            k = 10
            try:
                flow_eigvals, flow_eigvecs = np.linalg.eigh(a_f)
                dist_eigvals, dist_eigvecs = np.linalg.eigh(b_f)
                flow_top = flow_eigvecs[:, -k:]
                dist_top = dist_eigvecs[:, -k:]
                flow_centrality = np.abs(flow_top).sum(axis=1)
                dist_centrality = np.abs(dist_top).sum(axis=1)
                fac_order = np.argsort(-flow_centrality, kind='stable')
                loc_order = np.argsort(dist_centrality, kind='stable')
                p = np.empty(n, dtype=np.int64)
                p[fac_order] = loc_order
            except Exception:
                p = rng.permutation(n).astype(np.int64)
        else:
            p = rng.permutation(n).astype(np.int64)
        return p
    
    # Multiple initializations with ILS
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial solutions, including spectral methods
    n_init = 12
    for init_idx in range(n_init):
        if time.time() > deadline:
            break
        
        # Use different methods, with more weight on spectral
        if init_idx < 4:
            method = init_idx  # 0,1,2,3
        else:
            method = 4 + (init_idx % 2)  # 4,5 alternating
        
        p = make_initial_solution(method, rng)
        
        # Local search
        p, cost = local_search(p, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # ILS: perturb and search again if time permits
        if time.time() < deadline - 30:
            for ils_iter in range(5):
                if time.time() > deadline:
                    break
                # Perturb: random swaps
                n_swaps = rng.integers(5, 20)
                p_perturbed = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                p_perturbed, cost_p = local_search(p_perturbed, deadline)
                
                if cost_p < best_cost:
                    best_cost = cost_p
                    best_p = p_perturbed.copy()
                elif cost_p < cost:
                    p = p_perturbed.copy()
                    cost = cost_p
    
    # Final verification
    if best_p is None:
        best_p = make_initial_solution(0, rng)
    
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
