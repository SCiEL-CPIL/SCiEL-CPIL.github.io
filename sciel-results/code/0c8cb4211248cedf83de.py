"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """
    Solve the Quadratic Assignment Problem (QAP) for the given matrices a and b.
    Returns a permutation p of 0..n-1 that minimizes sum_{i,j} a[i,j] * b[p[i], p[j]].
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        # p is a permutation array of indices into b
        # Cost is sum(a[i,j] * b[p[i], p[j]])
        # This is trace(A * B_p) where B_p is permuted B
        b_sub = b_f[np.ix_(p, p)]
        return np.sum(a_f * b_sub)

    def get_delta(p, i, j):
        """
        Compute delta for swapping p[i] and p[j].
        p[i] = li, p[j] = lj.
        """
        li, lj = p[i], p[j]
        
        # Vectorized delta calculation
        # delta = dot(a[i]-a[j], b[lj,p]-b[li,p]) + dot(a[:,i]-a[:,j], b[p,lj]-b[p,li])
        
        diff_a_row = a_f[i] - a_f[j]
        diff_b_col = b_f[lj, p] - b_f[li, p]
        diff_a_col = a_f[:, i] - a_f[:, j]
        diff_b_row = b_f[p, lj] - b_f[p, li]
        
        return np.dot(diff_a_row, diff_b_col) + np.dot(diff_a_col, diff_b_row)

    def local_search_2opt(p, current_cost, time_limit):
        """
        Steepest descent 2-opt with vectorized delta.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Full scan for steepest descent
            for i in range(n):
                if time.time() > time_limit:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        
        return p, current_cost

    # --- Heuristic 1: Spectral Initialization ---
    def spectral_init():
        """
        Uses eigenvectors of B to align with flow structure.
        """
        # Symmetrize B if necessary for eigenvectors (though B is usually symmetric in QAP)
        # If B is not symmetric, we can use (B + B.T)/2 for spectral properties
        b_sym = (b_f + b_f.T) / 2.0
        
        # Get eigenvectors of B
        # We want to map high flow pairs to low distance pairs.
        # The cost is sum a[i,j] b[p[i], p[j]].
        # If we take the dominant eigenvector of B, it represents the "main axis" of distances.
        # We can align the "main axis" of A with it.
        
        # A simpler robust spectral method:
        # 1. Compute eigenvectors of B.
        # 2. Project A onto these eigenvectors?
        # Actually, a common spectral heuristic for QAP:
        # Let V be eigenvectors of B.
        # Let Y = A @ V.
        # Then p[i] is the argmax of Y[i, k] for the k-th eigenvector? No, that's not right.
        
        # Let's use the rank-matching approach but with spectral features.
        # Feature for facility i: projection of a[i, :] onto principal components of B?
        
        # Alternative:
        # 1. Compute eigenvectors of B (symmetric).
        # 2. Compute eigenvectors of A (symmetric part).
        # 3. Match the signs/magnitudes.
        
        # Let's stick to a robust spectral heuristic:
        # Compute eigenvectors of B.
        # For each facility i, compute the vector of its interactions projected onto the eigenvectors.
        
        # Simpler: Use the "barycentric" or "spectral" assignment.
        # Let's just use rank matching on row sums first, it's robust.
        # But the directive asks for GRASP. Let's implement GRASP construction.
        
        # For now, return a permutation based on rank matching to be safe.
        a_row_sums = a_f.sum(axis=1)
        b_row_sums = b_f.sum(axis=1)
        
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    # --- Heuristic 2: GRASP Construction ---
    def grasp_construction(alpha=0.5):
        """
        Greedy Randomized Adaptive Search Procedure.
        """
        p = np.full(n, -1, dtype=np.int64)
        assigned_facs = np.zeros(n, dtype=bool)
        assigned_locs = np.zeros(n, dtype=bool)
        
        # marginal[i][j] = cost increase if we assign facility i to location j
        marginal = np.zeros((n, n), dtype=np.float64)
        
        for _ in range(n):
            # Mask assigned
            mask = np.zeros((n, n), dtype=bool)
            mask[:, assigned_locs] = True
            mask[assigned_facs, :] = True
            
            # Get valid costs
            valid_costs = np.where(mask, np.inf, marginal)
            
            # Find min and max
            min_val = np.min(valid_costs)
            max_val = np.max(valid_costs[~mask]) if np.any(~mask) else 0
            
            # Threshold for Restricted Candidate List (RCL)
            # Typically: rcl_threshold = min + alpha * (max - min)
            # We want to pick from the best candidates.
            # If alpha is small, we are greedy. If alpha is large, we are random.
            # Let's use a dynamic alpha or fixed.
            
            rcl_threshold = min_val + alpha * (max_val - min_val)
            
            # Candidates are those with cost <= rcl_threshold
            candidates = np.where((valid_costs <= rcl_threshold) & (~mask))
            
            if len(candidates[0]) == 0:
                # Fallback to min
                idx = np.unravel_index(np.argmin(valid_costs), valid_costs.shape)
                i, j = idx
            else:
                # Randomly select from candidates
                # We can weight by inverse cost or just uniform.
                # Let's use uniform for simplicity.
                idx_choice = np.random.randint(len(candidates[0]))
                i = candidates[0][idx_choice]
                j = candidates[1][idx_choice]
            
            # Assign i to j
            p[i] = j
            assigned_facs[i] = True
            assigned_locs[j] = True
            
            # Update marginal
            # marginal += np.outer(a[:, i], b[j, :]) + np.outer(a[i, :], b[:, j])
            marginal += np.outer(a_f[:, i], b_f[j, :]) + np.outer(a_f[i, :], b_f_f[:, j])
            
        return p

    # --- Main Logic ---
    
    # 1. Generate initial solution using Rank Matching (robust baseline)
    p_init = spectral_init()
    current_cost = compute_cost(p_init)
    best_p = p_init.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial
    time_left_for_ls = deadline - 10.0
    p_ls, cost_ls = local_search_2opt(p_init.copy(), current_cost, time_left_for_ls)
    
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        
    # 3. GRASP Restarts
    # If time remains, do GRASP restarts.
    # Each restart: construct solution with GRASP, then local search.
    
    while time.time() < deadline - 5.0:
        # Construct new solution
        # Vary alpha slightly? Or keep fixed.
        alpha = 0.3  # Moderate randomness
        p_grasp = grasp_construction(alpha)
        
        cost_grasp = compute_cost(p_grasp)
        
        # If the constructed solution is already good, maybe skip LS? No, always LS.
        
        # Run local search
        time_left = deadline - 5.0 - time.time()
        if time_left < 1.0:
            break
            
        # Limit local search time per restart
        # If we have lots of time, allow longer LS.
        ls_time_limit = time.time() + min(time_left * 0.7, 20.0)
        
        p_r, cost_r = local_search_2opt(p_grasp, cost_grasp, ls_time_limit)
        
        if cost_r < best_cost:
            best_cost = cost_r
            best_p = p_r.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
