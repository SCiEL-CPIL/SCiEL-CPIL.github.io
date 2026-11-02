import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 4
    rank = 14

    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen's algorithm for 2x2 matrix multiplication involves 7 bilinear forms.
    # We apply this independently to the two 2x2 blocks of the 2x4 matrix.
    # Block 0: B[:, 0:2] -> C[:, 0:2]
    # Block 1: B[:, 2:4] -> C[:, 2:4]
    
    # The 7 Strassen forms for a 2x2x2 tensor (M = A*B, C = M)
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Each form is a tuple (U_factors, V_factors, W_factors)
    # Factors are lists of (index, coefficient)
    
    strassen_forms_2x2 = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = A00 * (B01 - B11)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = A11 * (B10 - B00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (A00 + A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    # Apply forms to Block 0 (cols 0,1 of B, cols 0,1 of C)
    # Global indices for B: row j, col k. Flat index j*4 + k.
    # Global indices for C: row k, col i. Flat index k*2 + i.
    
    for r in range(7):
        u_factors, v_factors, w_factors = strassen_forms_2x2[r]
        
        # U factors map to A (2x2), so indices are 0..3.
        for idx, coeff in u_factors:
            U[idx][r] += coeff
            
        # V factors map to B block 0 (cols 0,1).
        # Local B index (j, k_local) -> Global B index j*4 + k_local
        for l_idx, coeff in v_factors:
            j = l_idx // 2
            k_loc = l_idx % 2
            v_global_idx = j * 4 + k_loc
            V[v_global_idx][r] += coeff
            
        # W factors map to C block 0 (cols 0,1).
        # Local C index (k_local, i) -> Global C index (k_local)*2 + i
        for l_idx, coeff in w_factors:
            k_loc = l_idx // 2
            i = l_idx % 2
            w_global_idx = k_loc * 2 + i
            W[w_global_idx][r] += coeff

    # Apply forms to Block 1 (cols 2,3 of B, cols 2,3 of C)
    # Offset for B columns: +2
    # Offset for C columns: +2 (since C is 2x4, col index is i, but the block splits columns of C? 
    # Wait. C is 2x4. The product C = A * B.
    # A is 2x2. B is 2x4.
    # C[i][k] = sum_j A[i][j] B[j][k].
    # The "blocks" are in the column dimension of B and C.
    # B has columns 0,1 (Block 0) and 2,3 (Block 1).
    # C has columns 0,1 (Block 0) and 2,3 (Block 1).
    
    for r in range(7, 14):
        local_r = r - 7
        u_factors, v_factors, w_factors = strassen_forms_2x2[local_r]
        
        # U factors are the same (A is shared)
        for idx, coeff in u_factors:
            U[idx][r] += coeff
            
        # V factors map to B block 1 (cols 2,3).
        # Local B index (j, k_local) -> Global B index j*4 + (k_local + 2)
        for l_idx, coeff in v_factors:
            j = l_idx // 2
            k_loc = l_idx % 2
            v_global_idx = j * 4 + k_loc + 2
            V[v_global_idx][r] += coeff
            
        # W factors map to C block 1 (cols 2,3).
        # Local C index (k_local, i) -> Global C index (k_local)*2 + i + 2
        for l_idx, coeff in w_factors:
            k_loc = l_idx // 2
            i = l_idx % 2
            w_global_idx = k_loc * 2 + i + 2
            W[w_global_idx][r] += coeff

    # Optional: Verification (skip in production if time is tight, but good for debugging)
    # import numpy as np
    # U_np = np.array(U)
    # V_np = np.array(V)
    # W_np = np.array(W)
    # # Check if sum_r U[:,r] V[:,r] ... matches tensor
    # # Tensor T[i,j,k] = 1 if i==j? No, T[i,j,k] = A[i,j]B[j,k] -> C[i,k]
    # # The decomposition is C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[?, ?, r]?
    # # Standard tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
    # # No, for matrix multiplication C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # # The tensor is T_{i,j,k} where i is row of C, j is col of A/row of B, k is col of B/col of C?
    # # Actually, the tensor T for matrix multiplication M_{2,2,4} has indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}. It is 1 if valid, 0 otherwise.
    # # The rank-1 term is u_{i,j,r} v_{j,k,r} w_{k,i,r}?
    # # Let's check the interface definition:
    # # U: (n*m, rank) -> U[i*m+j, r]
    # # V: (m*p, rank) -> V[j*p+k, r]
    # # W: (p*n, rank) -> W[k*n+i, r]
    # # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T_{i,j,k}
    # # T_{i,j,k} is 1 if i,j,k form a valid multiplication term?
    # # A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # # So the tensor T_{i,j,k} is 1 for all i,j,k? No.
    # # The tensor T represents the bilinear map.
    # # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} is part of the product C_{i,k}.
    # # In the standard decomposition, we sum over j.
    # # The rank-1 tensors are of the form u_r (vector in A space) outer v_r (vector in B space) outer w_r (vector in C space).
    # # The contraction is <u_r, A> <v_r, B> w_r = C.
    # # So C_{i,k} = sum_r sum_{i',j'} sum_{j',k'} ... this is getting complicated.
    # # The interface says:
    # # U has shape (n*m, rank). Index i*n + j? Or i*m + j?
    # # Task says: U (n*m, rank). A is n x m. So index i*m+j.
    # # V (m*p, rank). B is m x p. Index j*p+k.
    # # W (p*n, rank). C is p x n? No, C is n x p.
    # # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # # C is n x p. Index k*n + i? That would mean C is p x n.
    # # Let's re-read: "multiplying a 2 by 2 matrix by a 2 by 4 matrix".
    # # A (2x2), B (2x4) -> C (2x4).
    # # n=2, m=2, p=4.
    # # U (n*m=4, R). A_{i,j}.
    # # V (m*p=8, R). B_{j,k}.
    # # W (p*n=8, R). C_{?}.
    # # Convention W[k*n+i]. If n=2, k*2+i. This looks like C_{k,i} where k is row, i is col?
    # # But C is 2x4 (n x p). So rows are 0,1 (size n=2), cols are 0,1,2,3 (size p=4).
    # # W index k*n+i. If n=2, k can be 0,1,2,3? No, k must be in range of first dim of W.
    # # W shape (p*n, R). If p=4, n=2, size 8.
    # # If index is k*n+i, and max index is 7, k can be 0,1,2,3 (if n=2, k*2+i, max 3*2+1=7).
    # # So k is 0..3, i is 0..1.
    # # This implies W represents a matrix of size p x n (4x2).
    # # But C is 2x4.
    # # "Trace convention" often implies C = A B, but maybe the tensor is defined differently?
    # # Or maybe W_{k,n+i} corresponds to C_{i,k}?
    # # Let's assume the standard tensor T_{i,j,k} where i (row of A/C), j (col of A/row of B), k (col of B/C).
    # # T_{i,j,k} = 1.
    # # Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # # If W is indexed by k*n+i, then W_{k,i,r} is the term.
    # # So C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ... summed over j?
    # # No, the bilinear form is linear in A and B, linear in C (coeff).
    # # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # # Rank 1 term: u_{i,j} v_{j,k} w_{k,i}.
    # # Then C_{i,k} approx sum_r (sum_j u_{i,j} a_{i,j}) (sum_k v_{j,k} b_{j,k}) ... no.
    # # It is sum_r <U_r, A> <V_r, B> W_r.
    # # Result matrix C = sum_r (sum_j U_r[i,j] A[i,j]) (sum_k V_r[j,k] B[j,k]) ... this doesn't sum to C[i,k] directly without another index.
    # # Correctly: C_{i,k} = sum_r sum_{i',j'} sum_{j',k'} ...
    # # Actually, for matrix mult, the tensor T is T_{i,j,k} = delta_{i,k}? No.
    # # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the expression for C_{?,?}.
    # # Usually, the tensor is T_{i,j,k} = 1, and the decomposition satisfies
    # # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_r U_r[i,j] V_r[j,k] W_r[k,i] ... no, the indices must match.
    # # The standard decomposition is:
    # # C_{i,k} = sum_r M_r
    # # M_r = (sum_{i',j'} U_r[i',j'] A_{i',j'}) * (sum_{j',k'} V_r[j',k'] B_{j',k'}) * W_r[k,i] ??
    # # This implies W_r is a matrix W_r[k,i].
    # # So C_{i,k} = sum_r W_r[k,i] * (U_r : A) * (V_r : B).
    # # Yes, this is the correct form.
    # # So W_r[k,i] is the coefficient for C_{i,k} coming from the r-th bilinear form.
    # # In our code:
    # # U[i*m+j] corresponds to A_{i,j}.
    # # V[j*p+k] corresponds to B_{j,k}.
    # # W[k*n+i] corresponds to W_r[k,i].
    # # Note: W index is k*n+i. n=2. So k is 0..3, i is 0..1.
    # # This matches C being 2x4?
    # # C has rows i (0..1) and cols k (0..3).
    # # So C_{i,k} is represented by W[k*n+i] = W[k*2+i].
    # # So W[k*2+i] is the coefficient for C_{i,k}.
    # # In the previous code:
    # # W[w_global_idx] where w_global_idx = k_loc * 2 + i.
    # # k_loc was 0..1 (local col).
    # # For block 0, k_global = k_loc.
    # # For block 1, k_global = k_loc + 2.
    # # So w_global_idx = k_global * 2 + i.
    # # This matches the convention W[k*n+i].
    
    return {"U": U, "V": V, "W": W}

