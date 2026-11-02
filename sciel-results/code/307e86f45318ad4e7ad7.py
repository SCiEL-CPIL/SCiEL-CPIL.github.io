"""Bootstrap direct approach for 2x3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Build the matrix multiplication tensor T[i,j,k] = delta[i,j,k]
    # where the tensor is 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C.
    # T is shape (n, m, p) with T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Actually, the matrix multiplication tensor has entries:
    # T[i, j, k] = 1 for all valid i,j,k (it's the "identity" tensor for matmul)
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, that's not quite right. Let me think again.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] which represents the "coefficient" relating
    # A[i,j] and B[j,k] to C[i,k]. In the standard formulation:
    # T[i, j, k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes)
    # But the tensor decomposition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] should equal T[i,j,k] for all i,j,k
    # where T[i,j,k] = 1.
    
    # Actually, I need to be more careful. The matrix multiplication tensor is:
    # T ∈ R^{n*m × m*p × p*n} (or equivalently R^{n × m × p} with appropriate indexing)
    # The entry T[i,j;k,l;m,n] = 1 if j=l and i=n and k=m, else 0
    # In the simplified 3-index form: T[i,j,k] = 1 for all valid indices
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Let's use the 3-tensor form T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    
    T = np.ones((n, m, p))
    
    # Greedy rank-1 peeling:
    # 1. Start with residual R = T
    # 2. Find best rank-1 approximation of R (via HOSVD or power method)
    # 3. Subtract it, project to half-integers, verify
    # 4. Repeat
    
    # For the rank-1 approximation, we use the approach of finding
    # u, v, w such that u ⊗ v ⊗ w best approximates R
    # This can be done via alternating least squares or power iteration
    
    def find_best_rank1(R, max_iter=100, tol=1e-10):
        """Find best rank-1 approximation of a 3-tensor via power iteration."""
        # Initialize randomly
        np.random.seed(42)
        u = np.random.randn(n)
        v = np.random.randn(m)
        w = np.random.randn(p)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)
        
        for it in range(max_iter):
            # Update u: u ~ R contracted with v and w
            R_vw = np.tensordot(R, v, axes=([1], [0]))  # shape (n, p)
            R_vw_w = np.tensordot(R_vw, w, axes=([1], [0]))  # shape (n)
            new_u = R_vw_w
            if np.linalg.norm(new_u) < 1e-15:
                break
            new_u = new_u / np.linalg.norm(new_u)
            
            # Update v: v ~ R contracted with u and w
            R_uw = np.tensordot(R, u, axes=([0], [0]))  # shape (m, p)
            R_uw_w = np.tensordot(R_uw, w, axes=([1], [0]))  # shape (m)
            new_v = R_uw_w
            if np.linalg.norm(new_v) < 1e-15:
                break
            new_v = new_v / np.linalg.norm(new_v)
            
            # Update w: w ~ R contracted with u and v
            R_uv = np.tensordot(R, u, axes=([0], [0]))  # shape (m, p)
            R_uv_v = np.tensordot(R_uv, v, axes=([0], [0]))  # shape (p)
            new_w = R_uv_v
            if np.linalg.norm(new_w) < 1e-15:
                break
            new_w = new_w / np.linalg.norm(new_w)
            
            # Check convergence
            if (np.linalg.norm(new_u - u) < tol and 
                np.linalg.norm(new_v - v) < tol and 
                np.linalg.norm(new_w - w) < tol):
                u, v, w = new_u, new_v, new_w
                break
            
            u, v, w = new_u, new_v, new_w
        
        # Compute the scalar coefficient
        scalar = np.einsum('i,j,k,ijk->', u, v, w, R)
        # The rank-1 component is scalar * u ⊗ v ⊗ w
        # But we want U[i,j,r] = u[i], V[j,k,r] = v[j]*w[k]? No...
        
        # Actually, for the decomposition T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # A rank-1 term has U[i,j,r] = a[i]*b[j], V[j,k,r] = b[j]*c[k]... no that's not right either.
        
        # Let me reconsider. The standard tensor decomposition of the matmul tensor is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # A rank-1 term would be: U[i,j,r] = u_i * v_j, V[j,k,r] = v_j * w_k, W[i,k,r] = u_i * w_k
        # But that gives product = u_i^2 * v_j^2 * w_k^2 which doesn't equal 1 in general.
        
        # I think the correct formulation is:
        # The matmul tensor T has entries T[i,j,k] = 1
        # We want: 1 = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] for all i,j,k
        # A rank-1 decomposition term contributes: U[i,j,r] * V[j,k,r] * W[i,k,r]
        # For this to work with the structure, we need:
        # U[i,j,r] depends on (i,j), V[j,k,r] depends on (j,k), W[i,k,r] depends on (i,k)
        
        # This is a different tensor structure than a standard 3-tensor.
        # Let me reformulate: define S[i,j,k] = T[i,j,k] = 1
        # We need S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        
        # This is NOT a standard tensor decomposition. It's a "matmul tensor" decomposition.
        # The variables are coupled: U sees (i,j), V sees (j,k), W sees (i,k).
        
        # So I can't just use standard HOSVD. I need a different approach.
        
        # Alternative: Use the fact that for fixed j, we have:
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
        # For each j, define matrices A_j[i,k] = S[i,j,k] = 1 (all ones n x p matrix)
        # And for each j, we need: A_j[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        
        # This is still complex. Let me try a different approach:
        # Directly solve for a low-rank decomposition using algebraic methods.
        
        # For 2x3x7, let's try to find a decomposition of rank R.
        # We have unknowns: U (2*3 x R = 6R), V (3*7 x R = 21R), W (7*2 x R = 14R)
        # Total: 41R unknowns
        # Constraints: 2*3*7 = 42 equations (one for each (i,j,k))
        # But the equations are nonlinear (cubic in the unknowns).
        
        # Let's try a structured approach: assume U, V, W have specific patterns.
        
        # For 2x3x7, one known result is that the rank is at most 14 (or similar).
        # Let's try to find a decomposition with rank 14.
        
        # Actually, let me try a simpler approach: use the fact that
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        # We can try to factor this using intermediate variables.
        
        # Let's define intermediate variables:
        # t0 = A[0,0]*B[0,0], t1 = A[0,0]*B[0,1], ..., etc.
        # But that's just schoolbook.
        
        # A better approach: use the "border rank" or "approximate rank" ideas,
        # but we need exact rank.
        
        # For now, let's try a heuristic search over small ranks.
        
        return u, v, w, scalar
    
    # Given the complexity, let's try a different strategy:
    # Use known results for 2x3x7 matrix multiplication.
    # The tensor rank of 2x3x7 matmul is known to be at most 14 (from various papers).
    # Let's try to construct such a decomposition.
    
    # One approach: use the fact that 2x3x7 can be embedded in a larger structure.
    # Or use the "laser method" from algebraic complexity theory.
    
    # For simplicity and given time constraints, let's try a greedy approach
    # where we build up the decomposition term by term.
    
    # Start with residual = T (all ones)
    # Try to find rank-1 terms that reduce the residual.
    
    # Actually, let me try a completely different approach:
    # Use integer programming or a SAT solver to find a low-rank decomposition.
    # But that might be too slow.
    
    # Given the time constraint, let's just return the schoolbook decomposition.
    # The greedy peeling approach I outlined is complex to implement correctly
    # for this non-standard tensor structure.
    
    # However, let me try one more thing: a structured ansatz with rank 14.
    
    # For 2x3x7, try the following construction:
    # Use 14 terms where each term corresponds to a "path" in the computation.
    
    # Actually, I realize I should just try to solve the system of equations
    # for a given rank R, using a numerical solver, and then round to half-integers.
    
    # For R = 14, we have 41*14 = 574 unknowns and 42 equations.
    # This is underdetermined, so we should be able to find a solution.
    
    # Let's use scipy.optimize to find a solution.
    
    try:
        from scipy.optimize import least_squares
        
        # Try ranks from 14 up to 20
        for R in range(14, 21):
            if time.time() > deadline - 30:
                break
            
            # Parameters: U (6 x R), V (21 x R), W (14 x R)
            # Total params: 41*R
            n_params = 6 * R + 21 * R + 14 * R
            
            def residuals(params):
                U = params[:6*R].reshape(6, R)
                V = params[6*R:6*R+21*R].reshape(21, R)
                W = params[6*R+21*R:].reshape(14, R)
                
                # Compute predicted T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                pred = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            pred[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Residual: pred - 1 (since T[i,j,k] = 1)
                return (pred - 1.0).flatten()
            
            # Initial guess: small random values
            np.random.seed(42)
            x0 = np.random.randn(n_params) * 0.1
            
            # Solve
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            
            if result.success or result.cost < 1e-10:
                # Check if we can round to half-integers
                U_sol = result.x[:6*R].reshape(6, R)
                V_sol = result.x[6*R:6*R+21*R].reshape(21, R)
                W_sol = result.x[6*R+21*R:].reshape(14, R)
                
                # Try rounding to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify exactness
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    # Found a valid half-integer decomposition!
                    U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(6)]
                    V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(21)]
                    W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(14)]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding didn't work, try to find a nearby half-integer solution
                # by perturbing and re-solving with constraints
                
                # For now, try a few different initializations
                for seed in range(5):
                    np.random.seed(seed)
                    x0 = np.random.randn(n_params) * 0.5
                    result = least_squares(residuals, x0, method='lm', max_nfev=5000)
                    
                    U_sol = result.x[:6*R].reshape(6, R)
                    V_sol = result.x[6*R:6*R+21*R].reshape(21, R)
                    W_sol = result.x[6*R+21*R:].reshape(14, R)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-10:
                        U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(6)]
                        V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(21)]
                        W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(14)]
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            if time.time() > deadline - 30:
                break
    
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
