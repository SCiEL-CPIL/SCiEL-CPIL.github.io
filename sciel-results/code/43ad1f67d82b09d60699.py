"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a decomposition with rank < 30 using rank-1 peeling
    # We'll use a more systematic approach with linear algebra
    
    def find_rank1_peeling(residual, max_trials=200, als_iters=30):
        """Find a good rank-1 approximation to the residual tensor."""
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        
        # Try multiple random initializations
        for trial in range(max_trials):
            np.random.seed(42 + trial)
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # ALS iterations
            for _ in range(als_iters):
                # Update u
                u = np.einsum('ijk,j,k->i', residual, v, w)
                # Update v
                v = np.einsum('ijk,i,k->j', residual, u, w)
                # Update w
                w = np.einsum('ijk,i,j->k', residual, u, v)
                
                # Normalize to prevent overflow
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                norm_w = np.linalg.norm(w)
                if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                    break
                u /= norm_u
                v /= norm_v
                w /= norm_w
            
            # Compute score
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        return best_score, best_u, best_v, best_w
    
    def try_rationalize(mat, max_denom=1000):
        """Try to find rational approximations for a matrix."""
        rationalized = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                frac = Fraction(val).limit_denominator(max_denom)
                row.append(float(frac))
            rationalized.append(row)
        return rationalized
    
    def verify_decomposition(U_frac, V_frac, W_frac, n, m, p, tol=1e-8):
        """Verify that the decomposition exactly reproduces the tensor."""
        U_arr = np.array(U_frac)
        V_arr = np.array(V_frac)
        W_arr = np.array(W_frac)
        
        # Check diagonal entries
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        
        # Check off-diagonal entries should be zero
        for i1 in range(n):
            for j1 in range(m):
                for k1 in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i1 != i2 or j1 != j2 or k1 != k2:
                                    val = np.sum(U_arr[i1*m+j1] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
                                    err = abs(val)
                                    max_err = max(max_err, err)
        
        return max_err < tol
    
    # Try to find a low-rank decomposition
    # Start with various target ranks
    for target_rank in range(20, 31):
        residual = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        success = True
        for rank_step in range(target_rank):
            # Find best rank-1 peeling
            score, u, v, w = find_rank1_peeling(residual, max_trials=100, als_iters=25)
            
            if score < 1e-8:
                success = False
                break
            
            # Subtract the rank-1 tensor
            residual -= np.einsum('i,j,k->ijk', u, v, w)
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            # Check if residual is close to zero
            if np.linalg.norm(residual) < 1e-10:
                break
        
        if not success:
            continue
        
        if np.linalg.norm(residual) > 1e-8:
            continue
        
        # Build factor matrices
        U_mat = np.array(U_list).T
        V_mat = np.array(V_list).T
        W_mat = np.array(W_list).T
        
        # Try to rationalize
        U_frac = try_rationalize(U_mat, max_denom=1000)
        V_frac = try_rationalize(V_mat, max_denom=1000)
        W_frac = try_rationalize(W_mat, max_denom=1000)
        
        # Verify
        if verify_decomposition(U_frac, V_frac, W_frac, n, m, p, tol=1e-6):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to schoolbook
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
