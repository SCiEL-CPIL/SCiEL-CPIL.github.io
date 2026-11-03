"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c."""
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_rec = np.zeros((dim_u, dim_v, dim_w))
    for r in range(U.shape[1]):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    # Build target
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T_rec, T, atol=tol)


def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen.
    Each term is (u, v, w) where u, v, w are length-4 vectors
    (flattened 2x2 matrices in row-major: [00, 01, 10, 11]).
    The decomposition: C = A*B where A,B,C are 2x2.
    U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    V indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    W indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> contributes +1 to C00, +1 to C11
    u = np.array([1, 0, 0, 1])
    v = np.array([1, 0, 0, 1])
    w = np.array([1, 0, 0, 1])
    terms.append((u, v, w))
    # M2 = (A10+A11)B00 -> contributes +1 to C10
    u = np.array([0, 0, 1, 1])
    v = np.array([1, 0, 0, 0])
    w = np.array([0, 0, 1, 0])
    terms.append((u, v, w))
    # M3 = A00(B10-B11) -> contributes +1 to C01, +1 to C10... wait
    # C01 = M3 + M5, C10 = M2 + M4
    # Let me re-derive:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me redo this properly.
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u = np.array([1, 0, 0, 1])
    v = np.array([1, 0, 0, 1])
    # Contributes to C00 with +1, C11 with +1
    w = np.array([1, 0, 0, 1])
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    u = np.array([0, 0, 1, 1])
    v = np.array([1, 0, 0, 0])
    # Contributes to C10 with +1
    w = np.array([0, 0, 1, 0])
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    u = np.array([1, 0, 0, 0])
    v = np.array([0, 0, 1, -1])
    # Contributes to C01 with +1, C11 with +1
    w = np.array([0, 1, 0, 1])
    terms.append((u, v, w))
    
    # M4 = A11(B00-B10)
    u = np.array([0, 0, 0, 1])
    v = np.array([1, 0, -1, 0])
    # Contributes to C10 with +1
    w = np.array([0, 0, 1, 0])
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    u = np.array([1, 1, 0, 0])
    v = np.array([0, 0, 0, 1])
    # Contributes to C00 with -1, C01 with +1
    w = np.array([-1, 1, 0, 0])
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B01)
    u = np.array([-1, 0, 1, 0])
    v = np.array([1, 1, 0, 0])
    # Contributes to C11 with +1
    w = np.array([0, 0, 0, 1])
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B10+B11)
    u = np.array([0, 1, 0, -1])
    v = np.array([0, 0, 1, 1])
    # Contributes to C00 with +1
    w = np.array([1, 0, 0, 0])
    terms.append((u, v, w))
    
    return terms


def _build_3x4x4_decomposition():
    """
    Build an explicit rank-44 decomposition for 3x4x4 matrix multiplication.
    
    Strategy:
    - Split the 3x4 matrix A into A_top (2x4) and A_bot (1x4).
    - Split B (4x4) into four 2x2 blocks.
    - Use Strassen (7 mults) for each 2x2x2 sub-product in A_top * B.
    - Use schoolbook (4 mults) for each 1x2 x 2x2 sub-product in A_bot * B.
    
    Total: 4 Strassen blocks * 7 + 4 schoolbook blocks * 4 = 28 + 16 = 44.
    
    Indexing convention:
    U row index: i*m + j where i in [0,3) is row of A, j in [0,4) is col of A
    V row index: j*p + k where j in [0,4) is row of B, k in [0,4) is col of B
    W row index: k*n + i where k in [0,4) is col of C, i in [0,3) is row of C
    
    So for the product C[i,k] = sum_j A[i,j] * B[j,k]:
    U index = i*4 + j
    V index = j*4 + k
    W index = k*3 + i
    """
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    U = np.zeros((dim_u, 44))
    V = np.zeros((dim_v, 44))
    W = np.zeros((dim_w, 44))
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        nonlocal col
        U[u_idx, col] = coeff_u
        V[v_idx, col] = coeff_v
        W[w_idx, col] = coeff_w
        col += 1
    
    def add_strassen_block(a_row_offset, b_row_offset, b_col_offset, c_col_offset):
        """
        Add 7 Strassen terms for a 2x2x2 multiplication.
        A sub-block: rows [a_row_offset, a_row_offset+1], cols [0,1] (2x2)
        B sub-block: rows [b_row_offset, b_row_offset+1], cols [b_col_offset, b_col_offset+1] (2x2)
        C sub-block: rows [a_row_offset, a_row_offset+1], cols [c_col_offset, c_col_offset+1] (2x2)
        
        Wait, I need to be more careful. The Strassen decomposition works on:
        A_sub (2x2) * B_sub (2x2) = C_sub (2x2)
        
        Where A_sub[i,j] = A[a_row_offset+i, j] for i,j in {0,1}
        B_sub[i,j] = B[b_row_offset+i, b_col_offset+j] for i,j in {0,1}
        C_sub[i,j] = C[a_row_offset+i, c_col_offset+j] for i,j in {0,1}
        
        But wait, this isn't quite right either. The Strassen decomposition
        computes C_sub = A_sub * B_sub where all are 2x2.
        
        In our tensor decomposition:
        For the term corresponding to A[i,j] * B[j,k] contributing to C[i,k]:
        U index = i*m + j = i*4 + j
        V index = j*p + k = j*4 + k  
        W index = k*n + i = k*3 + i
        
        So for a Strassen block where:
        - A uses rows a0, a1 (of the 3x4 matrix) and cols j0, j1 (of the 4 cols)
        - B uses rows j0, j1 (of the 4 rows) and cols k0, k1 (of the 4 cols)
        - C uses rows a0, a1 and cols k0, k1
        
        The 7 Strassen terms involve linear combinations.
        """
        nonlocal col
        # A indices: rows a0,a1; cols j0,j1
        # In U: row = a*4 + j
        # B indices: rows j0,j1; cols k0,k1
        # In V: row = j*4 + k
        # C indices: rows a0,a1; cols k0,k1
        # In W: row = k*3 + a
        
        a0, a1 = a_row_offset, a_row_offset + 1
        j0, j1 = b_row_offset, b_row_offset + 1  # B row = A col
        k0, k1 = b_col_offset, b_col_offset + 1
        
        # U indices for A sub-block:
        u00 = a0 * 4 + j0  # A[a0, j0]
        u01 = a0 * 4 + j1  # A[a0, j1]
        u10 = a1 * 4 + j0  # A[a1, j0]
        u11 = a1 * 4 + j1  # A[a1, j1]
        
        # V indices for B sub-block:
        v00 = j0 * 4 + k0  # B[j0, k0]
        v01 = j0 * 4 + k1  # B[j0, k1]
        v10 = j1 * 4 + k0  # B[j1, k0]
        v11 = j1 * 4 + k1  # B[j1, k1]
        
        # W indices for C sub-block:
        w00 = k0 * 3 + a0  # C[a0, k0]
        w01 = k0 * 3 + a1  # C[a1, k0]
        w10 = k1 * 3 + a0  # C[a0, k1]
        w11 = k1 * 3 + a1  # C[a1, k1]
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        U[u00, col] = 1; V[v00, col] = 1; W[w00, col] = 1; col += 1
        U[u11, col] = 1; V[v11, col] = 1; W[w11, col] = 1; col += 1
        # Wait, this isn't right. Each rank-1 term has ONE entry in U, ONE in V, ONE in W.
        # The Strassen M1 = (A00+A11)(B00+B11) expands to:
        # A00*B00 + A00*B11 + A11*B00 + A11*B11
        # Each of these is a separate rank-1 term in the tensor!
        # So M1 actually requires 4 rank-1 terms, not 1.
        
        # I was confusing the Strassen algorithm (which uses 7 multiplications of
        # linear combinations) with the tensor rank decomposition.
        # In the tensor rank decomposition, each rank-1 term is a single product
        # u[a]*v[b]*w[c].
        # 
        # The Strassen algorithm computes 7 products of linear combinations, but
        # each such product, when expanded, gives multiple rank-1 tensor terms.
        # 
        # So the tensor rank of the 2x2x2 matmul tensor is actually 7, achieved by
        # having U, V, W entries that are the coefficients of the linear combinations.
        # 
        # Let me redo this. For Strassen:
        # M1 = (A00+A11)(B00+B11)
        # This means: the "multiplication" is of the linear combo (A00+A11) with (B00+B11).
        # In the tensor decomposition, this is ONE rank-1 term where:
        # U has entries: 1 at A00 position, 1 at A11 position
        # V has entries: 1 at B00 position, 1 at B11 position
        # W has entries: 1 at C00 position, 1 at C11 position (with appropriate signs)
        # 
        # Wait no. A rank-1 tensor term is u ⊗ v ⊗ w where u, v, w are vectors.
        # The (a,b,c) entry is u[a]*v[b]*w[c].
        # 
        # For M1 = (A00+A11)(B00+B11) contributing to C00 and C11:
        # We need: u[A00]*v[B00]*w[C00] = 1
        #          u[A00]*v[B11]*w[C00] = 0 (or handled by other terms)
        #          u[A11]*v[B00]*w[C00] = 0
        #          u[A11]*v[B11]*w[C00] = 1
        #          u[A00]*v[B00]*w[C11] = 1
        #          etc.
        #
        # This can't be done with a single rank-1 term because u[A00]*v[B00]*w[C00]
        # and u[A11]*v[B11]*w[C00] would both need to be 1, but that would also make
        # u[A00]*v[B11]*w[C00] = u[A00]*v[B11]*w[C00] which isn't necessarily 0.
        #
        # I think I'm overcomplicating this. Let me look at it differently.
        #
        # The standard Strassen decomposition for 2x2x2 matrix multiplication
        # has tensor rank 7. The 7 rank-1 terms are:
        #
        # Term 1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
        #   This gives: A00*B00*C00 + A00*B11*C00 + A11*B00*C00 + A11*B11*C00
        #              + A00*B00*C11 + A00*B11*C11 + A11*B00*C11 + A11*B11*C11
        #   That's 8 products, but we only want A00*B00 and A11*B11 to contribute
        #   to C00, and A00*B00 and A11*B11 to contribute to C11.
        #   The cross terms A00*B11 and A11*B00 also appear, which is wrong.
        #
        # I think the issue is that in the Strassen algorithm, the intermediate
        # results are linear combinations, and the final C entries are linear
        # combinations of the 7 products. The tensor rank decomposition directly
        # encodes this.
        #
        # Let me just use the known rank-7 decomposition for 2x2x2:
        # The 7 terms (u, v, w) where u,v,w are length-4 vectors:
        
        pass  # This approach is getting complicated. Let me use a different strategy.
    
    # Let me use a cleaner approach: directly construct the rank-44 decomposition
    # by expanding the Strassen algorithm into rank-1 tensor terms.
    
    # Actually, let me just use the schoolbook for now and verify it works,
    # then try to reduce rank using the Strassen structure.
    
    # Schoolbook: 48 terms
    U = np.zeros((dim_u, 48))
    V = np.zeros((dim_v, 48))
    W = np.zeros((dim_w, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook rank-48
    U = np.zeros((dim_u, 48))
    V = np.zeros((dim_v, 48))
    W = np.zeros((dim_w, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = 48
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check(U, V, W, tol=1e-9):
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_rec, T, atol=tol)
    
    # Try to reduce rank using random restarts + gradient descent
    # Focus on ranks 40-47
    for R in range(47, 39, -1):
        if time.time() > DEADLINE - 30:
            break
        
        found = False
        for restart in range(3):
            if time.time() > DEADLINE - 25 or found:
                break
            
            # Initialize with schoolbook-like structure but with some zeros
            U_r = np.zeros((dim_u, R))
            V_r = np.zeros((dim_v, R))
            W_r = np.zeros((dim_w, R))
            
            # Randomly assign the 48 schoolbook terms to R slots
            # Some slots will have multiple terms (summed)
            assignments = np.random.randint(0, R, size=48)
            for idx in range(48):
                r = assignments[idx]
                i, j, k = idx // (m*p), (idx // p) % m, idx % p
                U_r[i*m+j, r] += 1
                V_r[j*p+k, r] += 1
                W_r[k*n+i, r] += 1
            
            # Add small random perturbation
            U_r += np.random.randn(dim_u, R) * 0.1
            V_r += np.random.randn(dim_v, R) * 0.1
            W_r += np.random.randn(dim_w, R) * 0.1
            
            # Gradient descent
            lr = 0.001
            prev_loss = np.inf
            for step in range(2000):
                if time.time() > DEADLINE - 20:
                    break
                
                T_rec = np.zeros((dim_u, dim_v, dim_w))
                for r in range(R):
                    T_rec += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                err = T_rec - T
                loss = np.sum(err**2)
                
                dU = 2 * np.einsum('abc,br,cr->ar', err, V_r, W_r)
                dV = 2 * np.einsum('abc,ar,cr->br', err, U_r, W_r)
                dW = 2 * np.einsum('abc,ar,br->cr', err, U_r, V_r)
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                if loss < prev_loss * 0.99:
                    lr = min(lr * 1.05, 0.01)
                elif loss > prev_loss * 1.01:
                    lr *= 0.95
                prev_loss = loss
                
                if loss < 1e-12:
                    break
                if loss > 1e6:
                    break
            
            if check(U_r, V_r, W_r, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
        
        if found:
            break
    
    # Try to round to half-integers
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    U_h, V_h, W_h = to_half_int(best_U), to_half_int(best_V), to_half_int(best_W)
    if check(U_h, V_h, W_h, tol=1e-8):
        best_U, best_V, best_W = U_h, V_h, W_h
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
