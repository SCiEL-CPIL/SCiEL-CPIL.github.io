import numpy as np
from fractions import Fraction
import time
import sys

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110 # 10s margin
    
    # Try to find rank 20 decomposition via 2x2x5 blocks
    # Target: Find U', V', W' for 2x2x5 with rank 10
    # U' shape (4, 10), V' shape (10, 10), W' shape (10, 4)
    # Eq: sum_k U'[i,j,k] V'[j,l,k] W'[l,i,k] = 1 for i in 0..1, j in 0..1, l in 0..4
    
    found = False
    best_rank = 40
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook
    def make_schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = make_schoolbook()
    best_U, best_V, best_W = U, V, W
    
    # Search for 2x2x5 rank 10
    # We will iterate over random sparse U, V
    # and solve for W
    
    import itertools
    
    def verify_225(U, V, W):
        # U (4, r), V (10, r), W (10, 4) -- wait, W is (p*n, r) = (10, r)
        # Indices: i in 0..1, j in 0..1, l in 0..4
        # U[i*2+j, k], V[j*5+l, k], W[l*2+i, k]
        r = U.shape[1]
        for i in range(2):
            for j in range(2):
                for l in range(5):
                    s = 0
                    for k in range(r):
                        s += U[i*2+j, k] * V[j*5+l, k] * W[l*2+i, k]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to find a solution
    # We can fix U and V to be random integers in {-1, 0, 1}
    # and solve for W using np.linalg.lstsq
    
    for attempt in range(500):
        if time.time() > deadline:
            break
            
        r = 10
        U_try = np.random.randint(-1, 2, (4, r))
        V_try = np.random.randint(-1, 2, (10, r))
        
        # Build linear system for W
        # Variables W[l*2+i, k] for l in 0..4, i in 0..1, k in 0..9
        # Total 10*10 = 100 variables
        # Equations: 2*2*5 = 20
        # M[eq_idx, w_idx] = U[i,j,k] * V[j,l,k]
        
        A = np.zeros((20, 100))
        b = np.ones(20)
        
        eq_idx = 0
        for i in range(2):
            for j in range(2):
                for l in range(5):
                    for k in range(r):
                        # W variable index: (l*2+i)*r + k
                        w_idx = (l*2+i)*r + k
                        A[eq_idx, w_idx] = U_try[i*2+j, k] * V_try[j*5+l, k]
                    eq_idx += 1
        
        try:
            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if W_vec is close to integers or half-integers
            W_test = np.round(W_vec * 2) / 2
            if np.allclose(A @ W_test, b, atol=1e-9):
                W_mat = W_test.reshape((10, r))
                if verify_225(U_try, V_try, W_mat):
                    # Found a rank 10 decomposition for 2x2x5
                    # Construct rank 20 for 2x4x5
                    U_full = np.zeros((8, 20))
                    V_full = np.zeros((20, 20))
                    W_full = np.zeros((10, 20))
                    
                    # Block 1: A1 (rows 0-1 of A, cols 0-1) * B1 (rows 0-1 of B)
                    # A1 indices: i in 0..1, j in 0..1
                    # B1 indices: j in 0..1, l in 0..4
                    # U_full[i, j, k] = U_try[i, j, k] for j in 0..1
                    # V_full[j, l, k] = V_try[j, l, k] for j in 0..1
                    # W_full[l, i, k] = W_mat[l, i, k]
                    
                    for i in range(2):
                        for j in range(2):
                            for k in range(10):
                                U_full[i*4+j, k] = U_try[i*2+j, k]
                                V_full[j*5+l, k] = V_try[j*5+l, k] # Wait, V index is j*5+l
                                # V_try is (10, 10). V_try[j*5+l, k]
                                # V_full is (20, 20). V_full[j*5+l, k]
                                # This is correct for j in 0..1
                                
                    # Wait, the loop above is wrong. I need to copy correctly.
                    # U_try (4, 10). U_full (8, 20).
                    # U_full[0:4, 0:10] = U_try
                    # U_full[4:8, 10:20] = U_try (for A2)
                    # V_try (10, 10). V_full (20, 20).
                    # V_full[0:10, 0:10] = V_try (for B1, j in 0..1 -> rows 0..9 of V_full? No)
                    # V index is j*p + l.
                    # For B1, j is 0,1. l is 0..4.
                    # So rows 0..9 of V_full correspond to j=0,1.
                    # V_full[0:10, 0:10] = V_try
                    # For B2, j is 2,3. l is 0..4.
                    # So rows 10..19 of V_full correspond to j=2,3.
                    # V_full[10:20, 10:20] = V_try (shifted indices?)
                    # V_try[j*5+l, k]. For B2, j is 2,3.
                    # We want V_full[j*5+l, k] for j in 2,3 to be V_try[(j-2)*5+l, k].
                    
                    # Let's do it explicitly.
                    for i in range(2):
                        for j in range(2):
                            for l in range(5):
                                for k in range(10):
                                    # Block 1
                                    U_full[i*4+j, k] = U_try[i*2+j, k]
                                    V_full[j*5+l, k] = V_try[j*5+l, k]
                                    W_full[l*2+i, k] = W_mat[l*2+i, k]
                                    
                                    # Block 2
                                    # A2: i in 0..1, j in 2..3
                                    # B2: j in 2..3, l in 0..4
                                    # U_full[i*4+(j+2), 10+k] = U_try[i*2+j, k]
                                    # V_full[(j+2)*5+l, 10+k] = V_try[j*5+l, k]
                                    # W_full[l*2+i, 10+k] = W_mat[l*2+i, k]
                                    
                                    U_full[i*4+j+2, 10+k] = U_try[i*2+j, k]
                                    V_full[(j+2)*5+l, 10+k] = V_try[j*5+l, k]
                                    W_full[l*2+i, 10+k] = W_mat[l*2+i, k]
                    
                    # Verify full decomposition
                    def verify_full(U, V, W):
                        r = U.shape[1]
                        for i in range(2):
                            for j in range(4):
                                for l in range(5):
                                    s = 0
                                    for k in range(r):
                                        s += U[i*4+j, k] * V[j*5+l, k] * W[l*2+i, k]
                                    if abs(s - 1) > 1e-9:
                                        return False
                        return True
                    
                    if verify_full(U_full, V_full, W_full):
                        best_U, best_V, best_W = U_full, V_full, W_full
                        best_rank = 20
                        found = True
                        break
                        
        except Exception as e:
            pass
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
