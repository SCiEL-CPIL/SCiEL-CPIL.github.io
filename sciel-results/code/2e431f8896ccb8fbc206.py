"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best = {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-8):
        n, m, p = DIMENSIONS
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for i1 in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i1==i and k2==k and j1==j2) else 0.0
                                actual = np.dot(W[k*n+i, :], U[i1*m+j1, :] * V[j2*p+k2, :])
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    np.random.seed(42)
    
    # Try a few ranks
    for r in [36, 48, 60, 72]:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U
                U_new = np.zeros_like(U)
                for i1 in range(n):
                    for j1 in range(m):
                        coeffs = []
                        rhs = []
                        for i in range(n):
                            for k in range(p):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        expected = 1.0 if (i1==i and k2==k and j1==j2) else 0.0
                                        c = V[j2*p+k2, :] * W[k*n+i, :]
                                        coeffs.append(c)
                                        rhs.append(expected)
                        A_sys = np.array(coeffs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, np.array(rhs), rcond=None)
                        U_new[i1*m+j1, :] = sol
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j2 in range(m):
                    for k2 in range(p):
                        coeffs = []
                        rhs = []
                        for i in range(n):
                            for k in range(p):
                                for i1 in range(n):
                                    for j1 in range(m):
                                        expected = 1.0 if (i1==i and k2==k and j1==j2) else 0.0
                                        c = U[i1*m+j1, :] * W[k*n+i, :]
                                        coeffs.append(c)
                                        rhs.append(expected)
                        A_sys = np.array(coeffs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, np.array(rhs), rcond=None)
                        V_new[j2*p+k2, :] = sol
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        coeffs = []
                        rhs = []
                        for i1 in range(n):
                            for j1 in range(m):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        expected = 1.0 if (i1==i and k2==k and j1==j2) else 0.0
                                        c = U[i1*m+j1, :] * V[j2*p+k2, :]
                                        coeffs.append(c)
                                        rhs.append(expected)
                        A_sys = np.array(coeffs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, np.array(rhs), rcond=None)
                        W_new[k*n+i, :] = sol
                
                U, V, W = U_new, V_new, W_new
                
                if verify(U, V, W):
                    if r < best_rank:
                        best_rank = r
                        best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    break
        
        if best_rank < rank_school:
            break
    
    return best
# EVOLVE-BLOCK-END
