import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # The tensor T has shape (n, m, p)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that T_{i,j,k} = 1.
    # The decomposition is sum_{c} U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    # Note the indices: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The entry T[i,j,k] corresponds to the product of U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    # We want to find rank r such that there exist U, V, W with entries in Q (or R)
    # satisfying the equation.
    
    # Let's try to find a decomposition of rank r < rank_schoolbook.
    # We will use a randomized search.
    # For a fixed rank r, we can try to solve for U, V, W.
    # This is a hard problem. A common heuristic is to fix W and solve for U, V.
    # If W is fixed, the equation is linear in U and V? No, it's bilinear in U and V.
    # However, if we fix V, it's linear in U and W.
    
    # Let's use a simpler verification: 
    # If we find U, V, W, we check if U * V * W (in tensor sense) equals T.
    
    # Since finding a rank-44 decomposition for 3x3x5 is not trivial and might not exist,
    # we will attempt a few random searches for rank 44. If they fail, we return rank 45.
    
    def verify(U, V, W, r):
        # Check if sum_c U[:,c] * V[:,c] * W[:,c] == T
        # T[i,j,k] = 1
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We compute the tensor product sum.
        # For speed, we can use numpy.
        # T_computed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is expensive to do naively for every check.
        # We can precompute the structure.
        
        # Let's just do a spot check or full check if r is small.
        # Full check cost: r * n * m * p operations.
        # For r=44, n*m*p=45. 44*45 = 1980 ops. Very fast.
        
        T_comp = np.zeros((n, m, p))
        for c in range(r):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        T_comp[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
        
        # T_target is all 1s
        # Check if T_comp is close to 1s
        if not np.allclose(T_comp, 1.0, atol=1e-9):
            return False
        return True

    def try_find_decomposition(r, num_tries=5000):
        # Randomized search with integer/half-integer bias
        # We sample U and V from a small integer domain to increase the chance
        # that the resulting W (if solvable) is also rational/integer.
        
        for _ in range(num_tries):
            if time.time() > deadline:
                return None
            
            # Sparse random U, V with entries in {-1, 0, 1}
            # This mimics the structure of the schoolbook solution
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Construct the linear system for W
            # Decoupled by (i, k)
            W_sol = np.zeros((p * n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build A_ik (m x r)
                    # A_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # We can vectorize this for speed
                    # U_slice = U[i*m : (i+1)*m, :] shape (m, r)
                    # V_slice = V[:, k] ... wait V is (m*p, r).
                    # V[j*p+k, c] for fixed k, varying j.
                    # Indices for V: j*p + k for j in 0..m-1.
                    # This is not a contiguous slice if p != 1.
                    # But we can just index it.
                    
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik * w_vec = 1
                    # w_vec is W[k*n+i, :]
                    try:
                        # Use lstsq
                        w_vec, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check residual
                        if np.linalg.norm(A_ik @ w_vec - np.ones(m)) > 1e-9:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w_vec
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if verify(U, V, W_sol, r):
                    return U, V, W_sol
        
        return None

    # Try ranks from 44 down to 30
    # We focus heavily on 44 as it is the most likely candidate for improvement
    # over schoolbook (45).
    for r in [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        # Allocate more time/tries to higher ranks (closer to 45)
        if r == 44:
            tries = 2000
        elif r == 43:
            tries = 1000
        else:
            tries = 500
            
        result = try_find_decomposition(r, num_tries=tries)
        if result is not None:
            U, V, W = result
            # Snap to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap, V_snap, W_snap, r):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            else:
                # If snapping fails, try rounding to integers?
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, r):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
