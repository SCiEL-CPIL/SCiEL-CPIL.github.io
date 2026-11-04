"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank = n*m*p = 30
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n*m)]
        V = [[0] * schoolbook_rank for _ in range(m*p)]
        W = [[0] * schoolbook_rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition exactly reproduces the tensor."""
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        # Check off-diagonal entries are zero
        for i1 in range(n):
            for j1 in range(m):
                for k1 in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    continue
                                idx_u = i1*m + j1
                                idx_v = j2*p + k2
                                idx_w = k2*n + i2
                                val = np.sum(U_arr[idx_u] * V_arr[idx_v] * W_arr[idx_w])
                                err = abs(val)
                                max_err = max(max_err, err)
        return max_err
    
    def try_rationalize(mat, max_denom=8):
        """Try to find rational approximations for a matrix. Returns (rational_matrix, success)."""
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if abs(val) < 1e-10:
                    row.append(0)
                    continue
                # Try to find a rational with small denominator
                frac = Fraction(val).limit_denominator(max_denom)
                if abs(float(frac) - val) < 1e-8:
                    # Check if it's an integer or half-integer
                    if frac.denominator in [1, 2]:
                        row.append(frac)
                    else:
                        return None, False
                else:
                    return None, False
            result.append(row)
        return result, True
    
    def als_rank1(residual, n_iter=30, n_trials=20, seed=0):
        """Find best rank-1 approximation using ALS."""
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        np.random.seed(seed)
        for trial in range(n_trials):
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            for _ in range(n_iter):
                u = np.einsum('ijk,j,k->i', residual, v, w)
                v = np.einsum('ijk,i,k->j', residual, u, w)
                w = np.einsum('ijk,i,j->k', residual, u, v)
                # Normalize to prevent overflow
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)
                nw = np.linalg.norm(w)
                if nu > 1e-10: u /= nu
                if nv > 1e-10: v /= nv
                if nw > 1e-10: w /= nw
            
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        return best_u, best_v, best_w, best_score
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Use rank-1 peeling with rational verification at each step
    
    for target_rank in range(20, schoolbook_rank):
        residual = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        success = True
        for step in range(target_rank):
            # Find best rank-1 approximation
            u, v, w, score = als_rank1(residual, n_iter=30, n_trials=20, seed=step * 100)
            
            if score < 1e-10:
                # No more signal
                if step < target_rank - 1:
                    success = False
                    break
                else:
                    break
            
            # Try to rationalize
            U_mat = np.zeros((n*m, step + 1))
            V_mat = np.zeros((m*p, step + 1))
            W_mat = np.zeros((p*n, step + 1))
            
            # Fill in previous columns
            if step > 0:
                for c in range(step):
                    U_mat[:, c] = U_list[c]
                    V_mat[:, c] = V_list[c]
                    W_mat[:, c] = W_list[c]
            
            # Try to find rational values for the new column
            # We need to scale u, v, w so that <residual, u⊗v⊗w> matches
            # and the entries are rational
            # 
            # Strategy: try different scalings
            found = False
            for scale in [1.0, 0.5, 2.0, 0.25, 4.0, 1/3, 3, 2/3, 3/2]:
                u_scaled = u * scale
                v_scaled = v * (1.0/scale)
                w_scaled = w
                
                # Try to rationalize all three
                U_new, ok1 = try_rationalize(u_scaled.reshape(-1, 1), max_denom=8)
                V_new, ok2 = try_rationalize(v_scaled.reshape(-1, 1), max_denom=8)
                W_new, ok3 = try_rationalize(w_scaled.reshape(-1, 1), max_denom=8)
                
                if ok1 and ok2 and ok3:
                    # Verify this rank-1 term can be subtracted and residual remains valid
                    U_mat[:, step] = [float(x) for x in U_new[0]]
                    V_mat[:, step] = [float(x) for x in V_new[0]]
                    W_mat[:, step] = [float(x) for x in W_new[0]]
                    
                    # Update residual
                    residual -= np.einsum('i,j,k->ijk', U_mat[:, step], V_mat[:, step], W_mat[:, step])
                    
                    # Store
                    U_list.append(U_mat[:, step].copy())
                    V_list.append(V_mat[:, step].copy())
                    W_list.append(W_mat[:, step].copy())
                    found = True
                    break
            
            if not found:
                success = False
                break
            
            # Check if residual is close to zero
            if np.linalg.norm(residual) < 1e-10:
                if step < target_rank - 1:
                    # We finished early, which is fine
                    break
                else:
                    break
        
        if success and np.linalg.norm(residual) < 1e-8:
            # Verify the full decomposition
            U_final = np.array(U_list).T
            V_final = np.array(V_list).T
            W_final = np.array(W_list).T
            
            # Convert to lists of Fractions
            U_frac = []
            for i in range(n*m):
                row = []
                for c in range(len(U_list)):
                    val = U_final[i, c]
                    frac = Fraction(val).limit_denominator(8)
                    row.append(frac)
                U_frac.append(row)
            
            V_frac = []
            for i in range(m*p):
                row = []
                for c in range(len(V_list)):
                    val = V_final[i, c]
                    frac = Fraction(val).limit_denominator(8)
                    row.append(frac)
                V_frac.append(row)
            
            W_frac = []
            for i in range(p*n):
                row = []
                for c in range(len(W_list)):
                    val = W_final[i, c]
                    frac = Fraction(val).limit_denominator(8)
                    row.append(frac)
                W_frac.append(row)
            
            # Final verification
            err = verify_decomposition(U_frac, V_frac, W_frac)
            if err < 1e-6:
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fall back to schoolbook
    return build_schoolbook()
