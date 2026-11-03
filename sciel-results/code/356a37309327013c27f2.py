import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # The tensor identity for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
    # 
    # A rank-r decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t has entries U[i*m+j, t], v_t has entries V[j*p+k, t], w_t has entries W[k*n+i, t]
    #
    # The constraint is:
    # For all (i, j, k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # For all (i', k') ≠ (i, k): sum_t U[i'*m+j', t] * V[j'*p+k', t] * W[k*n+i, t] = 0
    # 
    # Wait, that's not quite right. Let me re-derive.
    #
    # C[i,k] = sum_t W[k*n+i, t] * (sum_{i',j'} U[i'*m+j', t] * A[i',j']) * (sum_{j',k'} V[j'*p+k', t] * B[j',k'])
    #
    # Expanding the product:
    # C[i,k] = sum_t W[k*n+i, t] * sum_{i',j',j',k'} U[i'*m+j', t] * V[j'*p+k', t] * A[i',j'] * B[j',k']
    #
    # For this to equal sum_j A[i,j] * B[j,k], we need the coefficient of A[i',j'] * B[j',k'] in C[i,k] to be:
    #   1 if i'=i and k'=k and j'=j (but j is the shared index, so j'=j is automatic when i'=i, k'=k)
    #   Actually, the product A[i',j'] * B[j',k'] contributes to C[i',k'] only.
    #   So the coefficient of A[i',j'] * B[j',k'] in C[i,k] should be:
    #     1 if (i',k') = (i,k)
    #     0 otherwise
    #
    # The coefficient is: sum_t W[k*n+i, t] * U[i'*m+j', t] * V[j'*p+k', t]
    #
    # So the constraint is:
    # For all (i', j', k', i, k):
    #   sum_t W[k*n+i, t] * U[i'*m+j', t] * V[j'*p+k', t] = 1 if i'=i and k'=k, else 0
    
    # This is a system of bilinear equations. Let's use an iterative approach:
    # 1. Initialize U, V, W randomly
    # 2. Fix two, solve for the third via least squares
    # 3. Iterate until convergence
    
    def compute_residual(U, V, W):
        """Compute the residual of the tensor identity."""
        res = 0.0
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            target = 1.0 if (i2 == i and k2 == k) else 0.0
                            val = 0.0
                            for t in range(rank):
                                val += W[k*n+i, t] * U[i2*m+j2, t] * V[j2*p+k2, t]
                            res += (val - target) ** 2
        return res
    
    def solve_for_W(U, V, rank):
        """Given U and V, solve for W using least squares."""
        W = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(p):
                # For each (i,k), we need to find W[k*n+i, :] such that:
                # For all (i2, j2, k2): W[k*n+i, :] @ (U[i2*m+j2, :] * V[j2*p+k2, :]) = target
                # where target = 1 if i2==i and k2==k, else 0
                A_eq = []
                b_eq = []
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            vec = U[i2*m+j2, :] * V[j2*p+k2, :]
                            target = 1.0 if (i2 == i and k2 == k) else 0.0
                            A_eq.append(vec)
                            b_eq.append(target)
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)
                sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                W[k*n+i, :] = sol
        return W
    
    def solve_for_U(V, W, rank):
        """Given V and W, solve for U using least squares."""
        U = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # For each (i,j), we need to find U[i*m+j, :] such that:
                # For all (i2, j2, k2, k, i): the constraint involves U[i2*m+j2, :]
                # Actually, U[i*m+j, :] appears in constraints where i2=i and j2=j
                # The constraint is: for all (i, k): sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] should contribute correctly
                # 
                # Let me think again. U[i*m+j, :] appears in the term:
                # sum_t W[k*n+i', t] * U[i'*m+j', t] * V[j'*p+k', t]
                # when i'=i and j'=j.
                # 
                # For fixed (i, j), the equations involving U[i*m+j, :] are:
                # For all (i', k', k): if i'=i (which it is), then:
                #   W[k*n+i, t] * U[i*m+j, t] * V[j*p+k', t] contributes to the coefficient
                #   of A[i,j] * B[j,k'] in C[i,k]
                #   This should be 1 if k'=k, else 0
                #
                # So for fixed (i, j):
                # For all (k, k'): sum_t U[i*m+j, t] * (W[k*n+i, t] * V[j*p+k', t]) = 1 if k'=k else 0
                # 
                # This gives p*p = 16 equations.
                A_eq = []
                b_eq = []
                for k in range(p):
                    for k2 in range(p):
                        vec = W[k*n+i, :] * V[j*p+k2, :]
                        target = 1.0 if (k2 == k) else 0.0
                        A_eq.append(vec)
                        b_eq.append(target)
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)
                sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                U[i*m+j, :] = sol
        return U
    
    def solve_for_V(U, W, rank):
        """Given U and W, solve for V using least squares."""
        V = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                # For fixed (j, k), V[j*p+k, :] appears in:
                # sum_t W[k'*n+i', t] * U[i'*m+j', t] * V[j'*p+k, t]
                # when j'=j.
                # 
                # For fixed (j, k):
                # For all (i, i2, k'): sum_t V[j*p+k, t] * (W[k'*n+i', t] * U[i'*m+j, t]) = 1 if i'=i and k'=k else 0
                # 
                # This gives n*n*p = 36 equations.
                A_eq = []
                b_eq = []
                for i in range(n):
                    for i2 in range(n):
                        for k2 in range(p):
                            vec = W[k2*n+i2, :] * U[i2*m+j, :]
                            target = 1.0 if (i2 == i and k2 == k) else 0.0
                            A_eq.append(vec)
                            b_eq.append(target)
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)
                sol, _, _, _ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                V[j*p+k, :] = sol
        return V
    
    # Try different ranks, starting from low
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    for rank in range(20, 37):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(rank * 1000 + 42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        best_res = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            
            # Alternating least squares
            W = solve_for_W(U, V, rank)
            U = solve_for_U(V, W, rank)
            V = solve_for_U(V, W, rank) if False else solve_for_V(U, W, rank)
            
            res = compute_residual(U, V, W)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res < 1e-10:
                break
        
        if best_res < 1e-8:
            # Found a valid decomposition
            best_rank = rank
            best_result = {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
