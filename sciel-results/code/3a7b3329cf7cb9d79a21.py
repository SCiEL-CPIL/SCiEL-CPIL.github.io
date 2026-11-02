import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal decomposition for 2x4 * 4x5 matrix multiplication.
    Starts with a Rank-36 Block-Strassen decomposition and uses numerical optimization
    to reduce the rank (try 35, 34, 33, 32) by minimizing the tensor reconstruction error.
    If a lower rank is found with near-zero error, it rounds to nearest half-integer
    and verifies. Returns the best valid decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build the target tensor T[i, j, k] = sum_l A[i, l] B[l, k]
    # The tensor entries are coefficients of A_ij * B_jk.
    # T is a 3D array of shape (n, m, p) where T[i, j, k] is 1 if the term A[i,j]*B[j,k] appears, else 0.
    # Actually, for matrix multiplication, the tensor is defined such that:
    # C = AB implies C_ik = sum_j A_ij B_jk.
    # The structure tensor T has entries T[i, j, k] = delta_{sum over j?} No.
    # The tensor is a sum of rank-1 tensors T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r].
    # The target tensor for matrix multiplication of (n x m) by (m x p) has non-zero entries at T[i, j, k] 
    # wait, the indices are (row of A, col of A/row of B, col of B).
    # The "standard" tensor has T[i, j, k] = 1 if the product A[i, j] * B[j, k] is part of the sum?
    # No, the tensor itself IS the set of products.
    # For MM, the tensor T has T[i, j, k] = 1 for all i, j, k? No.
    # Let's clarify: We want to compute C = A B.
    # C is linear in A and B.
    # The map (A, B) -> C is bilinear.
    # This can be represented by a tensor T such that C = A * T * B?
    # Usually, the decomposition is: C_{ik} = sum_{r=1}^R (sum_j U_{ij}^r A_{ij}) (sum_j V_{jk}^r B_{jk}) W_{ik}^r.
    # Let M_r be the multiplication step.
    # The condition is: sum_r U_{ij}^r V_{jk}^r W_{ik}^r = 1 * delta_{i,j,k}?
    # No. The equation is:
    # sum_{j} A_{ij} B_{jk} = C_{ik}.
    # We want to rewrite sum_j A_{ij} B_{jk} as a sum of R terms.
    # So, sum_j (1 * A_{ij} * 1 * B_{jk}) = C_{ik}.
    # The "target" tensor T_target has entries T_target[i, j, k] = 1.
    # Because the bilinear form is sum_{i,j,k} A_{ij} B_{jk} (with appropriate indices for C).
    # Wait, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T has T[i, j, k] = 1 for all i, j, k in the sum?
    # Yes, the tensor for standard multiplication has 1 at every (i, j, k) triple?
    # Let's check rank 1 term: A_{ij} B_{jk}. This contributes to C_{ik}.
    # If we have R=1, C_{ik} = (sum_j U_{ij} A_{ij}) (sum_j V_{jk} B_{jk}) W_{ik}.
    # We want this to equal sum_j A_{ij} B_{jk}.
    # This implies that the tensor of coefficients of A_{ij} B_{jk} in the expression must match.
    # In the R-term expansion, the coefficient of A_{ij} B_{jk} in C_{ik} is sum_r U_{ij}^r V_{jk}^r W_{ik}^r.
    # We require this coefficient to be 1 if the standard MM includes A_{ij} B_{jk} in C_{ik}.
    # Standard MM: C_{ik} = sum_j A_{ij} B_{jk}.
    # So, for fixed i, k, the coefficient of A_{ij} B_{jk} is 1.
    # Therefore, the target tensor T_target[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    T_target = np.ones((n, m, p))
    
    # Helper to build initial Rank-36 decomposition (Block-Strassen)
    def get_block_strassen_36():
        terms = []
        # Strassen 2x2x2 terms
        def add_strassen_2x2x2(i0, j0, k0):
            # 7 terms
            s = 1
            terms.append((
                {(i0, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1},
                {(i0+1, k0): 1, (i0+1, k0+1): -1}
            ))
            terms.append((
                {(i0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(i0, k0+1): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(i0, k0): 1, (i0+1, k0): 1}
            ))
            terms.append((
                {(i0, j0): 1, (i0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(i0, k0): -1, (i0, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0, j0+1): 1, (i0+1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1}
            ))
            
        def add_schoolbook_2x2x1(i0, j0, k0):
            # 4 terms
            terms.append(( {(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1} ))
            terms.append(( {(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1} ))

        # Block decomposition for 2x4 * 4x5
        # Split m=4 into 2+2.
        # Term 1: A[0:2, 0:2] * B[0:2, 0:5] (2x2 * 2x5)
        # Term 2: A[0:2, 2:4] * B[2:4, 0:5] (2x2 * 2x5)
        
        # We use Strassen for 2x2x2 parts and schoolbook for the rest?
        # Previous code:
        # add_strassen_2x2x2((0, 0), (0, 0), (0, 0)) -> handles A(0:2,0:2) * B(0:2,0:2)
        # add_strassen_2x2x2((0, 2), (2, 0), (0, 0)) -> handles A(0:2,2:4) * B(2:4,0:2)
        # add_strassen_2x2x2((0, 0), (0, 2), (0, 2)) -> handles A(0:2,0:2) * B(0:2,2:2)
        # add_strassen_2x2x2((0, 2), (2, 2), (0, 2)) -> handles A(0:2,2:4) * B(2:4,2:2)
        # This covers B cols 0:4.
        # Remaining B cols 4:5 (width 1).
        # add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4)) -> A(0:2,0:2) * B(0:2,4)
        # add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4)) -> A(0:2,2:4) * B(2:4,4)
        
        add_strassen_2x2x2(0, 0, 0)
        add_strassen_2x2x2(0, 2, 0)
        add_strassen_2x2x2(0, 0, 2)
        add_strassen_2x2x2(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return terms

    def terms_to_matrices(terms):
        R = len(terms)
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for r, (u_d, v_d, w_d) in enumerate(terms):
            for (i, j), val in u_d.items():
                U[i*m + j, r] += val
            for (j, k), val in v_d.items():
                V[j*p + k, r] += val
            for (i, k), val in w_d.items():
                W[k*n + i, r] += val
        return U, V, W

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We reshape and use einsum
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Sum over r
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Note: W is indexed by (k, i) in reshape, but matrix index was k*n+i.
        # So W_reshaped[k, i, r] corresponds to W[k*n+i, r]. Correct.
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    def error_metric(U_flat, target_shape_U, V_flat, target_shape_V, W_flat, target_shape_W):
        U = U_flat.reshape(target_shape_U)
        V = V_flat.reshape(target_shape_V)
        W = W_flat.reshape(target_shape_W)
        T_comp = compute_tensor(U, V, W)
        return np.sum((T_comp - T_target)**2)

    # Generate initial 36 terms
    init_terms = get_block_strassen_36()
    U0, V0, W0 = terms_to_matrices(init_terms)
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = 36
    best_err = 1e-6 # Should be 0
    
    # Try to reduce rank from 35 down to 30
    for target_rank in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize by taking the first target_rank columns of the 36-rank decomposition
        # This is a heuristic; sometimes random initialization is better, but we have limited time.
        U_init = U0[:, :target_rank]
        V_init = V0[:, :target_rank]
        W_init = W0[:, :target_rank]
        
        # Flatten
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        U_size = n * m * target_rank
        V_size = m * p * target_rank
        W_size = p * n * target_rank
        
        def objective(x):
            U = x[:U_size].reshape(n*m, target_rank)
            V = x[U_size:U_size+V_size].reshape(m*p, target_rank)
            W = x[U_size+V_size:].reshape(p*n, target_rank)
            return error_metric(U, (n*m, target_rank), V, (m*p, target_rank), W, (p*n, target_rank))
        
        try:
            # Use L-BFGS-B with gradient if available, or numerical gradient
            # scipy.optimize.minimize has method 'L-BFGS-B'
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 10000})
            
            if res.fun < 1e-8: # High precision convergence
                U_opt = res.x[:U_size].reshape(n*m, target_rank)
                V_opt = res.x[U_size:U_size+V_size].reshape(m*p, target_rank)
                W_opt = res.x[U_size+V_size:].reshape(p*n, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify rounded version
                T_rounded = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.allclose(T_rounded, T_target, atol=1e-9):
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    best_rank = target_rank
                    # We can break if we reach the theoretical lower bound or just keep going
                    # Lower bound for 2x4x5 is likely around 28-30.
                    
        except Exception:
            continue

    return best_result
# EVOLVE-BLOCK-END
