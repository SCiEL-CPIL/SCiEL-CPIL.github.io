"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 28
    
    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = jp*p + kp, c = kp*n + ip
    # T[a, b, c] = 1 if i==ip and j==jp and k==kp else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                a = i*m + j
                                b = jp*p + k
                                c = kp*n + ip
                                T[a, b, c] = 1.0
    
    # Compute matricization ranks to get lower bounds
    T_mat1 = T.reshape(n*m, m*p * p*n)
    lb1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
    
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    lb2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    lb3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
    
    lower_bound = max(lb1, lb2, lb3)
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try to find lower rank decompositions
    # Strategy: use random projections and solve for factors
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    np.random.seed(42)
    
    # Try ranks from lower_bound to schoolbook_rank - 1
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Multiple random initializations for this rank
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Iteratively refine: fix two factors, solve for the third
            # For fixed U, V: W[c, r] should satisfy
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This is a linear system for W given U, V
            # Similarly for other factors
            
            # Use alternating least squares style
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each c, we need W[c, :] such that
                # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all a,b
                # This is: (U * V^T) flattened... let's think differently
                # For fixed a, b: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # Let M_ab[c, r] = T[a,b,c] and N_ab[r] = U[a,r] * V[b,r]
                # We want: sum_r N_ab[r] W[c,r] = T[a,b,c]
                # This is a system of equations. For each c, W[c,:] is determined
                # by the constraints from all (a,b) pairs.
                
                # Better approach: treat as tensor decomposition via ALS
                # Fix U, V, solve for W:
                # T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
                # For each c, the equation is:
                # sum_r W[c,r] * (U[a,r] V[b,r]) = T[a,b,c] for all a,b
                # Let X_c[r, (a,b)] = W[c,r] * U[a,r] * V[b,r]... this is nonlinear
                
                # Simpler: use the fact that this is a linear system in W[c,:]
                # for each c, given U and V.
                # Build the system: for each (a,b), we have one equation
                # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # This is: (U * V^T)[a,b] dot W[c,:] = T[a,b,c]
                # But U*V^T is (n*m x m*p), and we need this for all c.
                
                # Actually, let's reshape. Define:
                # For fixed c, we want to find W_c = W[c,:] (length r)
                # such that for all a, b: sum_r U[a,r] V[b,r] W_c[r] = T[a,b,c]
                # Let B_c[a,b] = T[a,b,c] (shape n*m x m*p)
                # And let A[a,b,r] = U[a,r] V[b,r] (shape n*m x m*p x r)
                # We need: sum_r A[a,b,r] W_c[r] = B_c[a,b]
                # This is a linear system: (A reshaped) @ W_c = B_c reshaped
                
                A_reshaped = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        for rr in range(r):
                            A_reshaped[a * m*p + b, rr] = U[a, rr] * V[b, rr]
                
                B_reshaped = np.zeros((n*m * m*p, p*n))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_reshaped[a * m*p + b, c] = T[a, b, c]
                
                # Solve: A_reshaped @ W.T = B_reshaped
                # W has shape (p*n, r), so W.T has shape (r, p*n)
                # A_reshaped has shape (n*m*m*p, r)
                # We want: for each c, A_reshaped @ W[c,:] = B_reshaped[:, c]
                # So: A_reshaped @ W.T = B_reshaped
                # W.T = pinv(A_reshaped) @ B_reshaped
                
                try:
                    W_new = np.linalg.lstsq(A_reshaped, B_reshaped, rcond=None)[0].T
                except:
                    break
                
                # Update W
                W = W_new
                
                # Update U: fix V, W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: sum_r U[a,r] (V[b,r] W[c,r]) = T[a,b,c] for all b,c
                # Let C_a[b,c] = T[a,b,c]
                # Let D_a[b,c,r] = V[b,r] W[c,r]
                # sum_r U[a,r] D_a[b,c,r] = C_a[b,c]
                
                A_reshaped = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        for rr in range(r):
                            A_reshaped[b * p*n + c, rr] = V[b, rr] * W[c, rr]
                
                B_reshaped = np.zeros((m*p * p*n, n*m))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_reshaped[b * p*n + c, a] = T[a, b, c]
                
                try:
                    U_new = np.linalg.lstsq(A_reshaped, B_reshaped, rcond=None)[0].T
                except:
                    break
                
                U = U_new
                
                # Update V: fix U, W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: sum_r V[b,r] (U[a,r] W[c,r]) = T[a,b,c] for all a,c
                # Let E_b[a,c] = T[a,b,c]
                # Let F_b[a,c,r] = U[a,r] W[c,r]
                # sum_r V[b,r] F_b[a,c,r] = E_b[a,c]
                
                A_reshaped = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        for rr in range(r):
                            A_reshaped[a * p*n + c, rr] = U[a, rr] * W[c, rr]
                
                B_reshaped = np.zeros((n*m * p*n, m*p))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_reshaped[a * p*n + c, b] = T[a, b, c]
                
                try:
                    V_new = np.linalg.lstsq(A_reshaped, B_reshaped, rcond=None)[0].T
                except:
                    break
                
                V = V_new
            
            # Check if the decomposition works
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Found a valid decomposition of rank r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = r
                break  # No need to try higher ranks for this r
        
        if best_rank == r:
            # Found minimal rank for this r, can stop
            break
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
