"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from fractions import Fraction
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or at least valid) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Attempts to find a rank-83 decomposition via randomized search.
    Falls back to schoolbook rank 84 if no improvement is found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Schoolbook as baseline
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # Build the target tensor T
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Heuristic search for rank 83
    # We try to see if we can represent T with 83 terms.
    # This is a hard problem. We use a simple randomized restart approach with ALS (Alternating Least Squares).
    # If the residual norm drops to near 0, we have a candidate.
    
    def check_time():
        return time.time() > deadline

    def try_rank(R, max_iters=20):
        if check_time():
            return None
            
        # Initialize U, V, W randomly
        # To help convergence, we can start with the schoolbook decomposition and zero out one column,
        # then perturb.
        
        for attempt in range(max_iters):
            if check_time():
                return None
                
            # Initialize
            # Start with schoolbook, remove one term, and randomize the rest slightly?
            # Or just random small values.
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Alternatively, use a structured initialization
            # Let's try a few structured inits
            if attempt == 0:
                # Copy schoolbook, drop last col, add noise
                U = np.array(U_school[:n*m][:, :R])
                V = np.array(V_school[:m*p][:, :R])
                W = np.array(W_school[:p*n][:, :R])
                # Add small random perturbation to break symmetry
                U += np.random.randn(n * m, R) * 1e-4
                V += np.random.randn(m * p, R) * 1e-4
                W += np.random.randn(p * n, R) * 1e-4
            
            # ALS iterations
            for it in range(50):
                if check_time():
                    return None
                
                # Update U: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
                # For fixed V, W, U_{ij,r} can be solved via linear system?
                # Residual R_{ij} = T_{ij} - sum_r U_{ij,r} (V_{jk,r} W_{ki,r})
                # This is not a simple linear system for U because the coefficients depend on r.
                # However, if we fix r, it's linear.
                # Let's use a gradient step instead, which is simpler to implement correctly for trilinear.
                
                # Gradient of L = 0.5 * || T - sum_r U_r V_r W_r ||^2 w.r.t U
                # dL/dU_{ij,r} = - sum_{j,k} T_{ijk} V_{jk,r} W_{ki,r} + sum_{j,k} (sum_{r'} U_{ij,r'} V_{jk,r'} W_{ki,r'}) V_{jk,r} W_{ki,r}
                
                # This is O(R * n * m * p * (n+m+p)) roughly, which is small.
                
                # Compute current product P = sum_r U_r V_r W_r
                P = np.zeros((n, m, p), dtype=float)
                for r in range(R):
                    # u is (n*m, 1), v is (m*p, 1), w is (p*n, 1)
                    # Reshape u to (n, m, 1), v to (1, m, p), w to (p, 1, n) -> sum over k? No.
                    # T_{ijk} = U_{ij,r} V_{jk,r} W_{ki,r}
                    # u_r is vector of length n*m. index i*m+j.
                    # v_r is vector of length m*p. index j*p+k.
                    # w_r is vector of length p*n. index k*n+i.
                    
                    u_vec = U[:, r].reshape(n, m)
                    v_vec = V[:, r].reshape(m, p)
                    w_vec = W[:, r].reshape(p, n)
                    
                    # P_{ijk} += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
                    # This is a tensor product contraction.
                    # We can use np.einsum
                    # u[i,j] v[j,k] w[k,i]
                    P += np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
                
                # Check convergence
                err = np.linalg.norm(P - T)
                if err < 1e-8:
                    # Found a candidate!
                    # Verify exactly
                    return (U, V, W, R)
                
                # Compute gradients
                # dU_{ij,r} = - sum_{k} T_{ijk} v_vec[j,k] w_vec[k,i] + sum_{k} P_{ijk} v_vec[j,k] w_vec[k,i]
                # Let's compute term by term for each r
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(R):
                    u_vec = U[:, r].reshape(n, m)
                    v_vec = V[:, r].reshape(m, p)
                    w_vec = W[:, r].reshape(p, n)
                    
                    # Term 1: - sum_k T_{ijk} v[j,k] w[k,i]
                    # T is all 1s.
                    # t_v_w = einsum('jk,ki->ij', v_vec, w_vec)
                    t_v_w = np.einsum('jk,ki->ij', v_vec, w_vec)
                    
                    # Term 2: sum_k P_{ijk} v[j,k] w[k,i]
                    # p_v_w = einsum('ijk,jk,ki->ij', P, v_vec, w_vec)
                    p_v_w = np.einsum('ijk,jk,ki->ij', P, v_vec, w_vec)
                    
                    dU[:, r] = (p_v_w - t_v_w).flatten()
                    
                    # dV_{jk,r} = - sum_i T_{ijk} u[i,j] w[k,i] + sum_i P_{ijk} u[i,j] w[k,i]
                    # t_u_w = einsum('ij,ki->jk', u_vec, w_vec)
                    t_u_w = np.einsum('ij,ki->jk', u_vec, w_vec)
                    # p_u_w = einsum('ijk,ij,ki->jk', P, u_vec, w_vec)
                    p_u_w = np.einsum('ijk,ij,ki->jk', P, u_vec, w_vec)
                    
                    dV[:, r] = (p_u_w - t_u_w).flatten()
                    
                    # dW_{ki,r} = - sum_j T_{ijk} u[i,j] v[j,k] + sum_j P_{ijk} u[i,j] v[j,k]
                    # t_u_v = einsum('ij,jk->ki', u_vec, v_vec)
                    t_u_v = np.einsum('ij,jk->ki', u_vec, v_vec)
                    # p_u_v = einsum('ijk,ij,jk->ki', P, u_vec, v_vec)
                    p_u_v = np.einsum('ijk,ij,jk->ki', P, u_vec, v_vec)
                    
                    dW[:, r] = (p_u_v - t_u_v).flatten()
                
                # Gradient descent step
                lr = 0.1
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Optional: re-normalize or project? No, just let it flow.
                
                if err < 1e-10:
                     return (U, V, W, R)

        return None

    # Try to find rank 83
    # Given the complexity, this might not converge.
    # But it's worth a few seconds.
    result_83 = try_rank(83, max_iters=5)
    
    if result_83:
        U, V, W, R = result_83
        # Convert to lists of Fractions for exactness?
        # The prompt requires integer/half-integer entries.
        # If the solution is close to such values, we can round.
        # But if it's arbitrary floats, it's invalid.
        # We must check if entries are close to k/2.
        
        def is_valid_half_integer(arr, tol=1e-6):
            for val in arr:
                # Check if val is close to n/2
                n2 = round(val * 2)
                if abs(val - n2/2.0) > tol:
                    return False
            return True
        
        if is_valid_half_integer(U) and is_valid_half_integer(V) and is_valid_half_integer(W):
            # Convert to Fractions
            U_frac = [[Fraction(round(val*2), 2) for val in row] for row in U]
            V_frac = [[Fraction(round(val*2), 2) for val in row] for row in V]
            W_frac = [[Fraction(round(val*2), 2) for val in row] for row in W]
            
            # Final Verification with Fractions
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(R):
                            s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
            if valid:
                return {"U": U_frac, "V": V_frac, "W": W_frac}

    # Fallback to Schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
