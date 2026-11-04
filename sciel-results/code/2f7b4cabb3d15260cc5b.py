import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Ensure we're working with int64 for exact arithmetic in cost calc
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    def calc_cost(p):
        # Efficient cost calculation
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Using einsum or sum of products
        # b_perm[i,j] = b[p[i], p[j]]
        # cost = sum(a * b_perm)
        # This creates a full n x n array which is fine for n=256 (65k elements)
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    # --- Phase 0: Initialization ---
    # Spectral embedding for a strong initial guess
    try:
        A_sym = a + a.T
        B_sym = b + b.T

        # Use a moderate number of eigenvectors for alignment
        k_eig = min(16, n - 1)
        
        # eigh is stable for symmetric matrices
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)

        # Take eigenvectors with largest absolute eigenvalues
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]  # (n, k_eig)
        V_B = eigvecs_B[:, idx_B]  # (n, k_eig)

        # Build assignment cost: align spectral signatures
        # We want to maximize similarity between facility i's signature and location j's signature
        # Cost matrix C[i,j] should be low if they are similar.
        # Similarity ~ V_A[i,:] dot V_B[j,:]
        # So C = - V_A @ V_B.T
        C_spectral = -V_A @ V_B.T  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    except Exception:
        # Fallback to rank-based initialization
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute symmetric A for local search delta calculations
    # The objective is sum a[i,j] b[p[i],p[j]].
    # If we assume A and B are symmetric (common in QAP), we can simplify deltas.
    # However, to be safe and general, we use the full A and B in deltas if possible,
    # or assume symmetry if it holds.
    # For tai256c, A and B are symmetric.
    # Let's check symmetry once.
    is_sym_a = np.allclose(a, a.T)
    is_sym_b = np.allclose(b, b.T)
    
    # We will use a generic delta calculation that works for non-symmetric too,
    # but optimized for the symmetric case if true.
    # Actually, the vectorized delta in the inspiration relies on A_sym.
    # If A is not symmetric, the delta for swapping i and j is:
    # sum_k (a[i,k] + a[k,i]) * (b[p[j], p[k]] - b[p[i], p[k]]) + ... 
    # It's easier to just use the general formula:
    # Delta = sum_{k != i, j} (a[i,k] + a[k,i] + a[j,k] + a[k,j]) * (b[p[j], p[k]] - b[p[i], p[k]]) ... this gets messy.
    
    # Let's use the formula from the inspiration which assumes A_sym = A + A.T
    # and B_sym = B + B.T.
    # Objective: sum_{i,j} A[i,j] B[p[i],p[j]]
    # If A, B symmetric:
    # Cost = sum_{i<j} 2 A[i,j] B[p[i],p[j]] + sum_i A[i,i] B[p[i],p[i]]
    # Swapping i, j changes terms involving i or j.
    
    # Let's stick to a robust vectorized 2-opt that works generally.
    # D = B[np.ix_(p,p)]
    # Cost = sum(A * D)
    # Swap p[i], p[j]:
    # D_new differs in row/col i and j.
    # Delta = sum_k (A[i,k] + A[k,i]) * (D[j,k] - D[i,k]) + sum_k (A[j,k] + A[k,j]) * (D[i,k] - D[j,k])
    #       + (A[i,i] + A[j,j]) * (D[j,j] - D[i,i]) + (A[i,j] + A[j,i]) * (D[j,i] - D[i,j])
    # Wait, the cross terms A[i,j]D[i,j] and A[j,i]D[j,i] are handled in the sums if we are careful.
    
    # Let's derive properly.
    # C = sum_{u,v} A[u,v] D[u,v]
    # After swap i<->j:
    # D'[i,i] = D[j,j]
    # D'[j,j] = D[i,i]
    # D'[i,k] = D[j,k] for k!=i,j
    # D'[j,k] = D[i,k] for k!=i,j
    # D'[k,i] = D[k,j] for k!=i,j
    # D'[k,j] = D[k,i] for k!=i,j
    # D'[i,j] = D[j,i]
    # D'[j,i] = D[i,j]
    
    # Delta = A[i,i](D[j,j]-D[i,i]) + A[j,j](D[i,i]-D[j,j])
    #       + A[i,j](D[j,i]-D[i,j]) + A[j,i](D[i,j]-D[j,i])
    #       + sum_{k!=i,j} A[i,k](D[j,k]-D[i,k]) + A[j,k](D[i,k]-D[j,k])
    #       + sum_{k!=i,j} A[k,i](D[k,j]-D[k,i]) + A[k,j](D[k,i]-D[k,j])
    
    # Grouping terms for vectorization over j (swapping i with j):
    # Let i be fixed.
    # Term 1 (self i): A[i,i] D[j,j] - A[i,i] D[i,i]
    # Term 2 (self j): A[j,j] D[i,i] - A[j,j] D[j,j]
    # Term 3 (cross i,j): A[i,j] D[j,i] - A[i,j] D[i,j]
    # Term 4 (cross j,i): A[j,i] D[i,j] - A[j,i] D[j,i]
    # Term 5 (row i, col k): sum_{k!=i,j} A[i,k] D[j,k] - sum_{k!=i,j} A[i,k] D[i,k]
    # Term 6 (row j, col k): sum_{k!=i,j} A[j,k] D[i,k] - sum_{k!=i,j} A[j,k] D[j,k]
    # Term 7 (row k, col i): sum_{k!=i,j} A[k,i] D[k,j] - sum_{k!=i,j} A[k,i] D[k,i]
    # Term 8 (row k, col j): sum_{k!=i,j} A[k,j] D[k,i] - sum_{k!=i,j} A[k,j] D[k,j]

    # We can precompute for each i:
    # R_i[k] = A[i,k]
    # C_i[k] = A[k,i]
    # Then for a candidate j:
    # Sum5 = dot(A[i, :], D[j, :]) - dot(A[i, :], D[i, :]) - A[i,j]D[j,j] - A[i,i]D[i,i] (remove k=j, k=i)
    # This is getting complicated to vectorize across all j for a fixed i without loops.
    
    # The inspiration used a specific symmetric simplification.
    # Given tai256c is symmetric, let's assume A, B symmetric to use the faster path.
    # If not symmetric, we fall back to a slower but correct method or just use the symmetric approximation if close.
    
    if is_sym_a and is_sym_b:
        A_sym = a # already symmetric
        B_sym = b
        
        def local_search_sym(p, current_cost, time_limit):
            """
            Optimized 2-opt local search for symmetric QAP.
            """
            n = len(p)
            p = p.copy()
            D = b[np.ix_(p, p)].copy() # D[i,j] = b[p[i], p[j]]
            current_cost = int(np.sum(a * D))
            
            while time.time() < time_limit:
                improved = False
                for i in range(n - 1):
                    if time.time() > time_limit:
                        return p, current_cost
                    
                    js = np.arange(i + 1, n)
                    
                    # Vectorized delta calculation for symmetric case
                    # Delta(i, j) = sum_{k != i, j} (A[i,k] + A[j,k]) * (D[j,k] - D[i,k]) 
                    #               + (A[i,i] - A[j,j]) * (D[j,j] - D[i,i]) 
                    #               + (A[i,j] - A[j,i]) * (D[j,i] - D[i,j])  # Note A[i,j]=A[j,i], D[i,j]=D[j,i] so this is 0?
                    #               Wait, D is symmetric if B is symmetric.
                    #               If A, B symmetric, D is symmetric. A is symmetric.
                    #               Term 3+4: A[i,j](D[j,i]-D[i,j]) + A[j,i](D[i,j]-D[j,i]) = 0.
                    #               So cross term vanishes?
                    #               Let's re-verify.
                    #               Old: A[i,j]D[i,j] + A[j,i]D[j,i] = 2 A[i,j] D[i,j]
                    #               New: A[i,j]D[j,i] + A[j,i]D[i,j] = 2 A[i,j] D[i,j] (since D[i,j]=D[j,i])
                    #               Yes, cross term is unchanged.
                    
                    # So Delta = sum_{k != i, j} (A[i,k] + A[j,k]) * (D[j,k] - D[i,k]) 
                    #           + (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
                    
                    # Let S = sum_{k} (A[i,k] + A[j,k]) * (D[j,k] - D[i,k])
                    # S = sum_k A[i,k]D[j,k] - sum_k A[i,k]D[i,k] + sum_k A[j,k]D[j,k] - sum_k A[j,k]D[i,k]
                    # S = dot(A[i,:], D[j,:]) - dot(A[i,:], D[i,:]) + dot(A[j,:], D[j,:]) - dot(A[j,:], D[i,:])
                    
                    # We need to subtract k=i and k=j terms from S.
                    # k=i term in S: (A[i,i] + A[j,i]) * (D[j,i] - D[i,i])
                    # k=j term in S: (A[i,j] + A[j,j]) * (D[j,j] - D[i,j])
                    
                    # Delta = S - term_i - term_j + (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
                    
                    # Vectorize over js:
                    # A_i = A[i, :]
                    # D_i = D[i, :]
                    # D_js = D[js, :]
                    # A_js = A[js, :]
                    
                    # dot(A_i, D_js) -> (len(js),)
                    # dot(A_i, D_i) -> scalar
                    # dot(A_js, D_js) -> (len(js),)  # This is sum_k A[j,k] D[j,k] for each j
                    # dot(A_js, D_i) -> (len(js),)
                    
                    A_i = A_sym[i, :]
                    D_i = D[i, :]
                    D_js = D[js, :]
                    A_js = A_sym[js, :]
                    
                    # S_components
                    c1 = A_js @ D_i  # sum_k A[j,k] D[i,k] for each j in js -> shape (len(js),)
                    c2 = A_js @ D_js # This is element-wise dot? No, A_js is (m, n), D_js is (m, n).
                                     # We want sum_k A[j,k] D[j,k] for each j.
                                     # This is np.sum(A_js * D_js, axis=1)
                    c2 = np.sum(A_js * D_js, axis=1)
                    c3 = A_i @ D_js.T # sum_k A[i,k] D[j,k] for each j. A_i (n,), D_js.T (n, m). Result (m,)
                    c4 = np.dot(A_i, D_i) # scalar
                    
                    S = c3 - c4 + c2 - c1
                    
                    # Corrections
                    # term_i = (A[i,i] + A[j,i]) * (D[j,i] - D[i,i])
                    # A[i,i] is scalar. A[j,i] is A_js[:, i]. D[j,i] is D_js[:, i]. D[i,i] is scalar.
                    term_i = (A_sym[i, i] + A_js[:, i]) * (D_js[:, i] - D[i, i])
                    
                    # term_j = (A[i,j] + A[j,j]) * (D[j,j] - D[i,j])
                    # A[i,j] is A_i[js]. A[j,j] is np.diag(A_js). D[j,j] is np.diag(D_js). D[i,j] is D_i[js].
                    term_j = (A_i[js] + np.diag(A_js)) * (np.diag(D_js) - D_i[js])
                    
                    # Self term correction: (A[i,i] - A[j,j]) * (D[j,j] - D[i,i])
                    self_term = (A_sym[i, i] - np.diag(A_js)) * (np.diag(D_js) - D[i, i])
                    
                    deltas = S - term_i - term_j + self_term
                    
                    min_idx = np.argmin(deltas)
                    min_delta = deltas[min_idx]
                    
                    if min_delta < 0:
                        j = js[min_idx]
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        
                        # Update D
                        # D[i] and D[j] swap
                        D[i], D[j] = D[j].copy(), D[i].copy()
                        # D[:, i] and D[:, j] swap
                        D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                        
                        current_cost += int(min_delta)
                        improved = True
            
            if not improved:
                break
            
            return p, current_cost

    else:
        # Fallback for non-symmetric (slower)
        def local_search_gen(p, current_cost, time_limit):
            n = len(p)
            p = p.copy()
            current_cost = int(current_cost)
            start = time.time()
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for i in range(n - 1):
                    if time.time() > time_limit:
                        return p, current_cost
                    pi = p[i]
                    for j in range(i + 1, n):
                        if time.time() > time_limit:
                            return p, current_cost
                        pj = p[j]
                        
                        # Calculate delta
                        # Mask out i, j
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        mask[j] = False
                        pk = p[mask]
                        
                        delta = 0.0
                        delta += a[i, i] * (b[pj, pj] - b[pi, pi])
                        delta += a[j, j] * (b[pi, pi] - b[pj, pj])
                        delta += a[i, j] * (b[pj, pi] - b[pi, pj])
                        delta += a[j, i] * (b[pi, pj] - b[pj, pi])
                        delta += np.sum(a[i, mask] * (b[pj, pk] - b[pi, pk]))
                        delta += np.sum(a[mask, i] * (b[pk, pj] - b[pk, pi]))
                        delta += np.sum(a[j, mask] * (b[pi, pk] - b[pj, pk]))
                        delta += np.sum(a[mask, j] * (b[pk, pi] - b[pk, pj]))
                        
                        if delta < 0:
                            p[i], p[j] = p[j], p[i]
                            current_cost += int(delta)
                            improved = True
                            break # Restart j loop
            return p, current_cost

    # Select local search
    if is_sym_a and is_sym_b:
        local_search_func = local_search_sym
    else:
        local_search_func = local_search_gen

    # --- Phase 1: Initial Local Search ---
    p, current_cost = local_search_func(p, current_cost, time_limit=start_time + 15.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # --- Phase 2: Iterated Local Search with Spectral/Hungarian Perturbations ---
    # We will use a mix of:
    # 1. Random k-swaps
    # 2. Linear Assignment (Hungarian) on linearized cost matrix
    
    T = 1000.0
    min_T = 10.0
    
    while time.time() < deadline - 5.0:
        # Adaptive perturbation
        # Start with small perturbations, increase if stuck
        # For simplicity, let's alternate or use a fixed small number
        perturb_k = 2
        
        # Option A: Random perturbation
        p_pert = best_p.copy()
        for _ in range(perturb_k):
            i, j = np.random.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Option B: Linear Assignment Perturbation
        # Compute dynamic cost matrix C
        # C[i,j] = sum_k a[i,k] b[j,p[k]] + sum_k a[k,i] b[p[k],j]
        # This is the gradient of the linearization
        b_p = b[:, p] # b_p[j, k] = b[j, p[k]]
        b_p_T = b[p, :] # b_p_T[k, j] = b[p[k], j]
        
        C = a @ b_p.T + a.T @ b_p
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(C)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        
        # Decide which perturbation to use
        # Maybe 50/50? Or based on time?
        # Let's try LAP perturbation as it is more structural.
        p_cur = p_lap
        
        # Run local search on the perturbed solution
        time_budget = 5.0
        p_cur, cost_cur = local_search_func(p_cur, calc_cost(p_cur), time_limit=time.time() + time_budget)
        
        if cost_cur < best_cost:
            best_cost = cost_cur
            best_p = p_cur.copy()
            T = max(min_T, T * 0.9)
        else:
            T = max(min_T, T * 0.95)

    return best_p.tolist()
