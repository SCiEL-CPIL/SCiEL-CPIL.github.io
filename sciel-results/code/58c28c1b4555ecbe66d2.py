import numpy as np
import time
import sys
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110  # 10s margin

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, tol=1e-9):
        r = U.shape[1]
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_k U[i*m+j, k] * V[j*p+l, k] * W[l*n+i, k] == 1
        # Note: W index in problem is W[k*n+i, r] where k is the output column (l in our loop)
        # So W[l*n+i, k]
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0.0
                    for k in range(r):
                        s += U[i * m + j, k] * V[j * p + l, k] * W[l * n + i, k]
                    if abs(s - 1) > tol:
                        return False
        return True

    # Strategy: Find rank 20 decomposition for 2x4x5
    # This is equivalent to finding rank 10 decomposition for 2x2x5 (since 2x4x5 = 2 * 2x2x5)
    # Or more generally, we can search for rank R decomposition for 2x4x5 directly.
    # Let's try rank 20 first as it's the most likely improvement over 40.
    
    # The tensor T[i, j, l] = 1 for all i in 0..1, j in 0..3, l in 0..4.
    # We want U (8, R), V (20, R), W (10, R) such that
    # sum_k U[i*4+j, k] * V[j*5+l, k] * W[l*2+i, k] = 1
    
    # Approach:
    # 1. Fix U and V to be random sparse matrices (e.g., entries in {-1, 0, 1}).
    # 2. Construct the linear system for W.
    #    The equation is: sum_k (U[i,j,k] * V[j,l,k]) * W[l,i,k] = 1
    #    This is linear in W entries.
    #    Variables: W[l*2+i, k] for l in 0..4, i in 0..1, k in 0..R-1. Total 10*R variables.
    #    Equations: 2*4*5 = 40 equations.
    #    If R is large enough, we have an underdetermined system (if 10*R > 40, i.e., R > 4).
    #    We can solve for W using least squares and check if the solution is close to integers/half-integers.
    #    Actually, the problem statement says "integer/half-integer entries".
    #    Let's try to find integer solutions first, then half-integers.
    
    # Let's try R=20.
    R = 20
    np.random.seed(42)
    
    found = False
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        # Random sparse U, V
        # Using {-1, 0, 1} with some probability of 0
        U_try = np.random.choice([-1, 0, 1], size=(8, R))
        V_try = np.random.choice([-1, 0, 1], size=(20, R))
        
        # Build linear system A @ W_vec = b
        # A has shape (40, 10*R)
        # b is ones(40)
        A = np.zeros((40, 10 * R))
        b = np.ones(40)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    # Equation: sum_k U[i*m+j, k] * V[j*p+l, k] * W[l*n+i, k] = 1
                    for k in range(R):
                        coeff = U_try[i * m + j, k] * V_try[j * p + l, k]
                        if coeff != 0:
                            # W variable index: (l*n + i) * R + k
                            w_idx = (l * n + i) * R + k
                            A[eq_idx, w_idx] = coeff
                    eq_idx += 1
        
        # Solve A @ W_vec = b
        try:
            # Use lstsq to find minimum norm solution
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if W_vec is close to integers or half-integers
            # Try rounding to nearest integer
            W_int = np.round(W_vec)
            if np.allclose(A @ W_int, b, atol=1e-9):
                W_mat = W_int.reshape((p * n, R))
                if verify_full(U_try, V_try, W_mat):
                    best_U, best_V, best_W = U_try, V_try, W_mat
                    best_rank = R
                    found = True
                    break
                    
            # Try rounding to nearest half-integer
            W_half = np.round(W_vec * 2) / 2
            if np.allclose(A @ W_half, b, atol=1e-9):
                W_mat = W_half.reshape((p * n, R))
                if verify_full(U_try, V_try, W_mat):
                    best_U, best_V, best_W = U_try, V_try, W_mat
                    best_rank = R
                    found = True
                    break
                    
        except Exception as e:
            continue
            
    # If not found with R=20, try R=30? No, 20 is the target.
    # If R=20 fails, maybe try a different structure for U, V.
    # For example, block structure.
    
    if not found and time.time() < deadline - 10:
        # Try a structured approach:
        # Split m=4 into two blocks of 2.
        # This is the 2x2x5 approach from the current program.
        # Let's try to find rank 10 for 2x2x5.
        R_225 = 10
        np.random.seed(123)
        
        for attempt in range(2000):
            if time.time() > deadline:
                break
                
            U_225 = np.random.choice([-1, 0, 1], size=(4, R_225))
            V_225 = np.random.choice([-1, 0, 1], size=(10, R_225))
            
            A = np.zeros((20, 10 * R_225))
            b = np.ones(20)
            
            eq_idx = 0
            for i in range(2):
                for j in range(2):
                    for l in range(5):
                        for k in range(R_225):
                            coeff = U_225[i * 2 + j, k] * V_225[j * 5 + l, k]
                            if coeff != 0:
                                w_idx = (l * 2 + i) * R_225 + k
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
                        
            try:
                W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W_int = np.round(W_vec)
                if np.allclose(A @ W_int, b, atol=1e-9):
                    W_225 = W_int.reshape((10, R_225))
                    # Verify 2x2x5
                    ok = True
                    for i in range(2):
                        for j in range(2):
                            for l in range(5):
                                s = 0
                                for k in range(R_225):
                                    s += U_225[i * 2 + j, k] * V_225[j * 5 + l, k] * W_225[l * 2 + i, k]
                                if abs(s - 1) > 1e-9:
                                    ok = False
                                    break
                            if not ok: break
                        if not ok: break
                    if ok:
                        # Construct 2x4x5 from 2x2x5
                        U_full = np.zeros((8, 20))
                        V_full = np.zeros((20, 20))
                        W_full = np.zeros((10, 20))
                        
                        # Block 1: j in 0..1
                        for i in range(2):
                            for j in range(2):
                                for l in range(5):
                                    for k in range(10):
                                        U_full[i * 4 + j, k] = U_225[i * 2 + j, k]
                                        V_full[j * 5 + l, k] = V_225[j * 5 + l, k]
                                        W_full[l * 2 + i, k] = W_225[l * 2 + i, k]
                                        
                        # Block 2: j in 2..3
                        for i in range(2):
                            for j in range(2):
                                for l in range(5):
                                    for k in range(10):
                                        U_full[i * 4 + j + 2, 10 + k] = U_225[i * 2 + j, k]
                                        V_full[(j + 2) * 5 + l, 10 + k] = V_225[j * 5 + l, k]
                                        W_full[l * 2 + i, 10 + k] = W_225[l * 2 + i, k]
                                        
                        if verify_full(U_full, V_full, W_full):
                            best_U, best_V, best_W = U_full, V_full, W_full
                            best_rank = 20
                            found = True
                            break
            except Exception as e:
                continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
