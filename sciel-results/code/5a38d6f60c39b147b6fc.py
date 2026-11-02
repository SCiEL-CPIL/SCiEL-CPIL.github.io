import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _add_schoolbook_block(n_sub, m_sub, p_sub, i_off, j_off, k_off, U, V, W, c):
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                gi = i_off + i
                gj = j_off + j
                gk = k_off + k
                
                u_idx = gi * 5 + gj
                v_idx = gj * 6 + gk
                w_idx = gk * 3 + gi
                
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
                c += 1
    return c

def _add_strassen_block(i_off, j_off, k_off, U, V, W, c):
    # 2x2x2 block at offsets
    # Strassen coefficients
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need to map these to U, V, W columns.
    # U indices: (i, j)
    # V indices: (j, k)
    # W indices: (k, i)
    
    # A is 2x2, B is 2x2, C is 2x2.
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1
    # U: A00, A11 -> 1
    # V: B00, B11 -> 1
    # W: C00, C11 -> 1
    
    # This is not a simple rank 1. It's a sum of rank 1s?
    # No, Strassen uses 7 multiplications of sums.
    # Each multiplication is a "rank 1" in the tensor sense if we consider the factors?
    # Actually, the tensor decomposition is:
    # T = sum_t U_t \otimes V_t \otimes W_t
    # For Strassen, we have 7 terms.
    # Each term corresponds to one of the M's.
    # But M1 is (A00+A11)(B00+B11).
    # This is NOT a rank 1 tensor. It's a product of sums.
    # To express as sum of rank 1 tensors, we need to expand?
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # That would be 4 rank 1 terms.
    # Strassen's trick is that we don't expand, we compute the product of the sums.
    # But in tensor decomposition, we need to represent the multiplication tensor.
    # The multiplication tensor for 2x2x2 is sum_{i,j,k} e_i \otimes e_j \otimes e_k x_i y_j z_k.
    # Strassen's algorithm provides a decomposition into 7 rank 1 tensors?
    # No, Strassen's algorithm shows that the tensor can be decomposed into 7 terms of the form (linear form in A) \otimes (linear form in B) \otimes (linear form in C).
    # Yes! That is exactly what we need.
    
    # So we define 7 linear forms for U, V, W.
    
    # M1:
    # u: A00 + A11
    # v: B00 + B11
    # w: C00 + C11  (Wait, C00 = M1... so w is just the coefficient for C00? No.)
    
    # Let's look at the structure.
    # C = U * V * W^T ? No.
    # C_{i,k} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} ?
    # The task says: W[k*n+i, r].
    # So C_{i,k} = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # This is the standard form.
    
    # For Strassen M1:
    # Term 1 contributes to C00 and C11.
    # C00 += M1
    # C11 += M1
    # So W should have 1 at (0,0) and 1 at (1,1) for this term?
    # And U has 1 at (0,0) and (1,1).
    # And V has 1 at (0,0) and (1,1).
    # Let's check:
    # C00 = U00 V00 W00 + U01 V10 W10 + U10 V01 W01 + U11 V11 W11
    # If U=diag(1,1), V=diag(1,1), W=diag(1,1):
    # C00 = 1*1*1 + 0 + 0 + 0 = 1.
    # C11 = 0 + 0 + 0 + 1*1*1 = 1.
    # C01 = 0.
    # C10 = 0.
    # This gives C = diag(1,1).
    # But M1 = (A00+A11)(B00+B11).
    # If A=diag(1,0), B=diag(1,0), M1 = 1. C=diag(1,1). Correct.
    # If A=diag(0,1), B=diag(1,0), M1 = 0. C=0. Correct.
    # If A=diag(1,1), B=diag(1,0), M1 = 1. C=diag(1,1).
    # Actual C = A B = diag(1,0).
    # C00=1, C11=0.
    # But our term gives C11=1.
    # So M1 alone is not enough. We need to combine terms.
    
    # The Strassen formulas are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_t, we have a rank 1 tensor U_t \otimes V_t \otimes W_t.
    # And C is a linear combination of these.
    # Specifically, C_{i,k} = sum_t W_{t, i, k} (U_t \cdot A) (V_t \cdot B).
    # So W_{t, i, k} is the coefficient of M_t in C_{i,k}.
    
    # Let's define the 7 terms.
    
    # Term 1: M1
    # U: A00 + A11
    # V: B00 + B11
    # W: C00 + C11  (Coeffs: C00=1, C11=1)
    
    # Term 2: M2
    # U: A10 + A11
    # V: B00
    # W: C10 + C11  (Coeffs: C10=1, C11=-1)
    
    # Term 3: M3
    # U: A00
    # V: B10 - B11
    # W: C01 + C11  (Coeffs: C01=1, C11=1)
    
    # Term 4: M4
    # U: A11
    # V: B01 - B00
    # W: C00 + C10  (Coeffs: C00=1, C10=1)
    
    # Term 5: M5
    # U: A00 + A01
    # V: B11
    # W: C00 - C01  (Coeffs: C00=1, C01=-1)
    
    # Term 6: M6
    # U: A10 - A00
    # V: B00 + B01
    # W: C11  (Coeffs: C11=1)
    
    # Term 7: M7
    # U: A01 - A11
    # V: B10 + B11
    # W: C00  (Coeffs: C00=1)
    
    # We need to map these to the global indices.
    
    # U indices (local):
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # V indices (local):
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # W indices (local):
    # (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Term 1:
    # U: 1 at 0, 3
    # V: 1 at 0, 3
    # W: 1 at 0, 3
    
    # Term 2:
    # U: 1 at 2, 3
    # V: 1 at 0
    # W: 1 at 2, -1 at 3
    
    # Term 3:
    # U: 1 at 0
    # V: 1 at 2, -1 at 3
    # W: 1 at 1, 1 at 3
    
    # Term 4:
    # U: 1 at 3
    # V: 1 at 1, -1 at 0
    # W: 1 at 0, 1 at 2
    
    # Term 5:
    # U: 1 at 0, 1
    # V: 1 at 3
    # W: 1 at 0, -1 at 1
    
    # Term 6:
    # U: 1 at 2, -1 at 0
    # V: 1 at 0, 1
    # W: 1 at 3
    
    # Term 7:
    # U: 1 at 1, -1 at 3
    # V: 1 at 2, 3
    # W: 1 at 0
    
    # We will add these 7 columns.
    
    # Global U row indices for local (i,j):
    # gi = i_off + i
    # gj = j_off + j
    # u_idx = gi * 5 + gj
    
    # Global V row indices for local (j,k):
    # gj = j_off + j
    # gk = k_off + k
    # v_idx = gj * 6 + gk
    
    # Global W row indices for local (k,i):
    # gk = k_off + k
    # gi = i_off + i
    # w_idx = gk * 3 + gi
    
    # Term 1
    U[i_off*5 + j_off, c] = 1
    U[(i_off+1)*5 + (j_off+1), c] = 1
    V[j_off*6 + k_off, c] = 1
    V[(j_off+1)*6 + (k_off+1), c] = 1
    W[k_off*3 + i_off, c] = 1
    W[(k_off+1)*3 + (i_off+1), c] = 1
    c += 1
    
    # Term 2
    U[(i_off+1)*5 + (j_off+1), c] = 1
    U[(i_off+1)*5 + (j_off+2), c] = 1 # Wait, local (1,0) is i=1, j=0. Global j is j_off.
    # Correction:
    # Local (i,j) -> Global (i_off+i, j_off+j)
    # U index: (i_off+i)*5 + (j_off+j)
    
    # Term 2:
    # U: A10 (i=1, j=0), A11 (i=1, j=1)
    # V: B00 (j=0, k=0)
    # W: C10 (k=0, i=1), C11 (k=1, i=1)
    
    U[(i_off+1)*5 + j_off, c] = 1
    U[(i_off+1)*5 + (j_off+1), c] = 1
    V[j_off*6 + k_off, c] = 1
    W[k_off*3 + (i_off+1), c] = 1
    W[(k_off+1)*3 + (i_off+1), c] = -1
    c += 1
    
    # Term 3:
    # U: A00
    # V: B10 (j=1, k=0), B11 (j=1, k=1)
    # W: C01 (k=1, i=0), C11 (k=1, i=1)
    
    U[i_off*5 + j_off, c] = 1
    V[(j_off+1)*6 + k_off, c] = 1
    V[(j_off+1)*6 + (k_off+1), c] = -1
    W[(k_off+1)*3 + i_off, c] = 1
    W[(k_off+1)*3 + (i_off+1), c] = 1
    c += 1
    
    # Term 4:
    # U: A11
    # V: B01 (j=0, k=1), B00 (j=0, k=0)
    # W: C00 (k=0, i=0), C10 (k=0, i=1)
    
    U[(i_off+1)*5 + (j_off+1), c] = 1
    V[j_off*6 + (k_off+1), c] = 1
    V[j_off*6 + k_off, c] = -1
    W[k_off*3 + i_off, c] = 1
    W[k_off*3 + (i_off+1), c] = 1
    c += 1
    
    # Term 5:
    # U: A00, A01
    # V: B11
    # W: C00, C01
    
    U[i_off*5 + j_off, c] = 1
    U[i_off*5 + (j_off+1), c] = 1
    V[(j_off+1)*6 + (k_off+1), c] = 1
    W[k_off*3 + i_off, c] = 1
    W[(k_off+1)*3 + i_off, c] = -1
    c += 1
    
    # Term 6:
    # U: A10, A00
    # V: B00, B01
    # W: C11
    
    U[(i_off+1)*5 + j_off, c] = 1
    U[i_off*5 + j_off, c] = -1
    V[j_off*6 + k_off, c] = 1
    V[j_off*6 + (k_off+1), c] = 1
    W[(k_off+1)*3 + (i_off+1), c] = 1
    c += 1
    
    # Term 7:
    # U: A01, A11
    # V: B10, B11
    # W: C00
    
    U[i_off*5 + (j_off+1), c] = 1
    U[(i_off+1)*5 + (j_off+1), c] = -1
    V[(j_off+1)*6 + k_off, c] = 1
    V[(j_off+1)*6 + (k_off+1), c] = 1
    W[k_off*3 + i_off, c] = 1
    c += 1
    
    return c

def construct():
    start = time.time()
    deadline = start + 130.0
    
    n, m, p = DIMENSIONS
    
    # Allocate max possible rank (90)
    U = np.zeros((n * m, 90), dtype=object)
    V = np.zeros((m * p, 90), dtype=object)
    W = np.zeros((p * n, 90), dtype=object)
    
    c = 0
    
    # Partition:
    # n: [0], [1,2]
    # m: [0,1], [2,3], [4]
    # p: [0,1], [2,3], [4,5]
    
    n_splits = [(0, 1), (1, 3)] # (start, end) exclusive? No, let's use start, size.
    # Let's use (start, size)
    n_parts = [(0, 1), (1, 2)]
    m_parts = [(0, 2), (2, 2), (4, 1)]
    p_parts = [(0, 2), (2, 2), (4, 2)]
    
    for i_start, i_size in n_parts:
        for j_start, j_size in m_parts:
            for k_start, k_size in p_parts:
                if i_size == 1 and j_size == 2 and k_size == 2:
                    c = _add_schoolbook_block(1, 2, 2, i_start, j_start, k_start, U, V, W, c)
                elif i_size == 1 and j_size == 1 and k_size == 2:
                    c = _add_schoolbook_block(1, 1, 2, i_start, j_start, k_start, U, V, W, c)
                elif i_size == 2 and j_size == 2 and k_size == 2:
                    c = _add_strassen_block(i_start, j_start, k_start, U, V, W, c)
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    c = _add_schoolbook_block(2, 1, 2, i_start, j_start, k_start, U, V, W, c)
                else:
                    # Fallback schoolbook
                    c = _add_schoolbook_block(i_size, j_size, k_size, i_start, j_start, k_start, U, V, W, c)
    
    # Trim
    U = U[:, :c]
    V = V[:, :c]
    W = W[:, :c]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
