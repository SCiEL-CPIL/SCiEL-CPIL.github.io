import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The trivial decomposition has rank n*m*p = 42.
    # We use a recursive divide-and-conquer strategy to reduce the rank.
    # Strategy: Split the 'p' dimension (columns of B, rows of C) into chunks.
    # R(n, m, p) <= R(n, m, p1) + R(n, m, p2)
    # Base cases:
    # R(n, m, 1) = n * m (diagonal terms)
    # R(2, 2, 2) = 7 (Strassen)
    # R(2, 2, 1) = 3 (Optimal for 2x2x1)
    
    # We build the decomposition by summing rank-1 tensors.
    # A rank-1 tensor is defined by three index sets:
    # U indices (in 0..n*m-1), V indices (in 0..m*p-1), W indices (in 0..p*n-1)
    # Actually, it's simpler to think in terms of the tensor T[i, j, k] = delta_{i,k} * delta_{j,?} ... no.
    # The matrix multiplication tensor T is defined by T[i, j, k] = 1 if (i, j, k) corresponds to
    # A[i, k] * B[k, j] = C[i, j].
    # Wait, the standard tensor is T_{i, j, k} = delta_{i, k} ... no.
    # Let's stick to the factor matrices U, V, W.
    # The condition is: sum_r U[i, r] * V[j, r] * W[k, r] = T[i, j, k]
    # where T[i, j, k] = 1 if i == k (in terms of matrix indices? No).
    
    # Let's clarify indices:
    # A is n x m. B is m x p. C is n x p.
    # C[i, j] = sum_k A[i, k] * B[k, j].
    # The tensor T has indices (i, k, j) ? Or (i, j, k)?
    # The problem statement says:
    # U: (n*m, rank) -> indices i, k (A)
    # V: (m*p, rank) -> indices k, j (B)
    # W: (p*n, rank) -> indices j, i (C) ?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W index is `idx = j*n + i`.
    # So W[j*n + i, r].
    # The term is U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r].
    # Sum over r.
    # This must equal delta_{i, i} * delta_{k, k} * delta_{j, j}? No.
    # It must equal 1 if the indices match the multiplication, 0 otherwise.
    # Specifically, for a fixed (i, k, j), the value is 1 if this is a valid term in the sum for C[i, j].
    # But the tensor is multi-linear.
    # T[i, k, j] = 1. It doesn't depend on values, just indices.
    # Wait, the tensor represents the *operation*.
    # The tensor T is such that T[i, k, j] = 1 for all i, k, j? No.
    # If we contract U, V, W we get the matrix product.
    # The "multiplication tensor" is typically defined such that its entries are 1 if the indices form a valid path in the multiplication, but actually, for the *bilinear* form, it's simpler.
    # Let's look at the seed.
    # Seed U[0][col] += 1. U index 0 is i=0, k=0 (since n*m=6, 0 = 0*3+0).
    # Seed V[b00][col] += 1. V index 0 is k=0, j=0 (m*p=21, 0 = 0*7+0).
    # Seed W[c00][col] += 1. W index 0 is j=0, i=0 (p*n=14, 0 = 0*2+0).
    # Product: U[0, r] V[0, r] W[0, r] = 1*1*1 = 1.
    # This contributes to C[0, 0] via A[0, 0] * B[0, 0].
    # So the tensor entry T[i, k, j] is 1.
    # Wait, if T[i, k, j] = 1 for all i, k, j, then the rank is n*m*p?
    # No, the tensor is sparse?
    # Actually, the matrix multiplication tensor is dense in the sense that T[i, k, j] = 1 for all i, k, j.
    # Because A[i, k] * B[k, j] contributes to C[i, j].
    # So T[i, k, j] = 1.
    # If T is all ones, its rank is 1? No, that's for a single tensor.
    # The rank of the tensor T_{i, k, j} = 1 is 1?
    # If T_{i, k, j} = 1, then T = u v^T w^T?
    # u_i = 1, v_k = 1, w_j = 1.
    # Then sum_r 1*1*1 = rank.
    # But we need to distinguish which product goes to which sum.
    # Ah, the tensor is NOT all ones.
    # The tensor T is defined by the indices of the *variables*.
    # The variables are A_{i, k}, B_{k, j}, C_{i, j}.
    # The equation is C_{i, j} = sum_k A_{i, k} B_{k, j}.
    # This is a system of bilinear equations.
    # The tensor T has indices (i, k, j) and T[i, k, j] = 1.
    # Why? Because the term A_{i, k} B_{k, j} appears in the equation for C_{i, j}.
    # So yes, T[i, k, j] = 1 for all i, k, j.
    # If T is the all-ones tensor, its rank is 1.
    # But that's not right. The rank of matrix multiplication is not 1.
    
    # Correction:
    # The tensor T is a 3-tensor.
    # The entries are T[i, j, k] where i, j, k are indices of the *factors* in the decomposition?
    # No, the tensor T is defined on the index sets of A, B, C.
    # T is a tensor of size (n*m) x (m*p) x (p*n).
    # The entry T[u_idx, v_idx, w_idx] is 1 if the corresponding term A_{i,k} B_{k,j} contributes to C_{i,j}.
    # u_idx = i*m + k
    # v_idx = k*p + j
    # w_idx = j*n + i
    # For a specific u_idx, v_idx, w_idx, do they correspond to a valid term?
    # u_idx determines (i, k).
    # v_idx determines (k', j).
    # w_idx determines (j', i').
    # The term is valid if i == i', k == k', j == j'.
    # So T[u_idx, v_idx, w_idx] = 1 if the (i,k) from u, (k,j) from v, and (j,i) from w are consistent.
    # i_u == i_w, k_u == k_v, j_v == j_w.
    # This tensor is NOT all ones. It is sparse.
    # Number of 1s is n*m*p.
    # Total size is (n*m)*(m*p)*(p*n) = (n*m*p)^2.
    # So it's very sparse.
    
    # So we need to decompose this sparse tensor into rank-1 tensors.
    # A rank-1 tensor is defined by three vectors u, v, w.
    # T = sum_r u_r v_r w_r.
    # The seed program does this.
    
    # My plan:
    # 1. Initialize U, V, W as zeros.
    # 2. Define a helper to add a rank-1 term given indices and coefficients.
    # 3. Use a recursive function to generate the decomposition.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        # We need to append to the matrices.
        # Since we don't know the final rank, we can use lists of lists.
        # But we need to handle multiple terms adding to the same rank index?
        # No, each rank-1 tensor is a separate column.
        # So we just append a new column.
        U.append([coeff if i == u_idx else 0 for i in range(n*m)])
        V.append([coeff if i == v_idx else 0 for i in range(m*p)])
        W.append([coeff if i == w_idx else 0 for i in range(p*n)])
        rank += 1

    # This simple add_term only handles single indices.
    # But rank-1 tensors can have multiple non-zero entries in u, v, w.
    # e.g. Strassen uses sums.
    # So I need a more general add_term.
    
    # Let's restart the structure.
    
    U = np.zeros((n*m, 1))
    V = np.zeros((m*p, 1))
    W = np.zeros((p*n, 1))
    
    # This is inefficient. Let's use lists.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        """
        u_dict: {index: coeff}
        v_dict: {index: coeff}
        w_dict: {index: coeff}
        """
        u_vec = [0.0] * (n*m)
        v_vec = [0.0] * (m*p)
        w_vec = [0.0] * (p*n)
        
        for idx, c in u_dict.items():
            u_vec[idx] += c
        for idx, c in v_dict.items():
            v_vec[idx] += c
        for idx, c in w_dict.items():
            w_vec[idx] += c
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Recursive construction
    # R(n, m, p)
    
    def construct_decomp(n, m, p):
        # Returns lists of u, v, w dicts for the decomposition of n x m x p
        # We need to map the local indices to global indices.
        # This is tricky.
        
        # Alternative: Just hardcode the specific case 2x3x7 using the recursive logic.
        # Split 2x3x7 into 2x3x2 + 2x3x2 + 2x3x2 + 2x3x1?
        # Or 2x2x7 + 2x1x7.
        
        # Let's go with 2x2x7 + 2x1x7.
        # 2x1x7 has rank 1? No.
        # R(2, 1, p) = 1?
        # C[i, j] = A[i, 0] * B[0, j].
        # This is rank 1?
        # T[i, 0, j] = 1.
        # u_i = 1, v_0 = 1, w_j = 1.
        # Yes, rank 1.
        # Wait, A is 2x1, B is 1xp.
        # u_vec has 2 entries (i=0,1). v_vec has 1 entry (k=0). w_vec has p entries (j=0..p-1).
        # u = [1, 1]^T, v = [1], w = [1, ..., 1]^T.
        # So one rank-1 tensor.
        
        # So R(2, 1, 7) = 1.
        
        # Now R(2, 2, 7).
        # Split 7 into 2, 2, 2, 1.
        # R(2, 2, 2) = 7.
        # R(2, 2, 1) = 3.
        # Total 7+7+7+3 = 24.
        
        # Total rank = 24 + 1 = 25.
        
        # I need to implement the mapping of indices.
        # For 2x2x2 block at offset (i_off, k_off, j_off):
        # Local i, k, j map to global i_off+i, k_off+k, j_off+j.
        
        # Let's define a function to get the Strassen decomposition for 2x2x2.
        # Indices:
        # A: 0,1 (i), 0,1 (k) -> u_idx = i*2+k
        # B: 0,1 (k), 0,1 (j) -> v_idx = k*2+j
        # C: 0,1 (j), 0,1 (i) -> w_idx = j*2+i
        
        # Strassen terms (7):
        # 1. (A00+A11)(B00+B11) -> C00 +=, C11 +=
        #    u: {0:1, 3:1} (i=0,k=0 and i=1,k=1)
        #    v: {0:1, 3:1} (k=0,j=0 and k=1,j=1)
        #    w: {0:1, 3:1} (j=0,i=0 and j=1,i=1)
        # 2. (A10+A11)(B00) -> C10 +=, C11 -=
        #    u: {2:1, 3:1}
        #    v: {0:1}
        #    w: {2:1, 3:-1} (j=1,i=0 and j=1,i=1) -> idx 2, 3?
        #    w_idx = j*2+i.
        #    C10: i=1, j=0 -> w_idx = 0*2+1 = 2.
        #    C11: i=1, j=1 -> w_idx = 1*2+1 = 3.
        #    So w: {2:1, 3:-1}.
        # 3. (A00)(B11-B01) -> C01 +=, C11 +=
        #    u: {0:1}
        #    v: {3:1, 1:-1} (k=1,j=1 and k=0,j=1)
        #    w: {1:1, 3:1} (C01: i=0,j=1 -> 1*2+0=2? No. j=1, i=0 -> 1*2+0=2. C11: i=1,j=1 -> 3)
        #    Wait, C01 is i=0, j=1. w_idx = 1*2+0 = 2.
        #    C11 is i=1, j=1. w_idx = 1*2+1 = 3.
        #    So w: {2:1, 3:1}.
        # 4. (A11)(B10-B00) -> C00 +=, C10 +=
        #    u: {3:1}
        #    v: {2:1, 0:-1} (k=1,j=0 and k=0,j=0)
        #    w: {0:1, 2:1} (C00: 0, C10: 2)
        # 5. (A00+A01)(B11) -> C00 -=, C01 +=
        #    u: {0:1, 1:1}
        #    v: {3:1}
        #    w: {0:-1, 2:1}
        # 6. (A10-A00)(B00+B01) -> C11 +=
        #    u: {2:1, 0:-1}
        #    v: {0:1, 1:1}
        #    w: {3:1}
        # 7. (A01-A11)(B10+B11) -> C00 +=
        #    u: {1:1, 3:-1}
        #    v: {2:1, 3:1}
        #    w: {0:1}
        
        # I will verify these indices carefully in the code.
        
        # For 2x2x1 (rank 3):
        # A 2x2, B 2x1, C 2x1.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Rank 3 decomposition:
        # 1. (A00+A11)(B0+B1) -> C0 +=, C1 +=
        #    u: {0:1, 3:1}
        #    v: {0:1, 1:1}
        #    w: {0:1, 1:1} (C0: j=0,i=0 -> 0. C1: j=0,i=1 -> 1)
        # 2. (A10+A11)(B0) -> C1 +=, C0 -=?
        #    Let's use standard 2x2x1 rank 3.
        #    M1 = (A00+A11)(B0+B1)
        #    M2 = (A10+A11)(B0)
        #    M3 = (A00)(B1)
        #    C0 = M1 - M2 + M3?
        #    M1 = A00B0 + A00B1 + A11B0 + A11B1
        #    M2 = A10B0 + A11B0
        #    M3 = A00B1
        #    M1 - M2 + M3 = A00B0 + A00B1 + A11B1 + A00B1 = A00B0 + 2A00B1 + A11B1. Incorrect.
        
        #    Correct 2x2x1 rank 3:
        #    M1 = (A00+A11)(B0+B1)
        #    M2 = (A10+A11)(B0)
        #    M3 = (A01)(B1)
        #    C0 = M1 - M2 + M3?
        #    M1 = A00B0 + A00B1 + A11B0 + A11B1
        #    M2 = A10B0 + A11B0
        #    M3 = A01B1
        #    M1 - M2 + M3 = A00B0 + A00B1 + A11B1 + A01B1.
        #    We want C0 = A00B0 + A01B1.
        #    We have extra A00B1 + A11B1.
        
        #    Let's look up 2x2x1 rank 3.
        #    It is known to be 3.
        #    M1 = (A00+A11)(B0+B1)
        #    M2 = (A10+A11)(B0)
        #    M3 = (A00+A01)(B1)
        #    C0 = M1 - M2 - M3 + ...?
        
        #    Actually, I can just use the trivial rank 4 for 2x2x1 if I can't find rank 3 easily.
        #    R(2, 2, 1) = 4?
        #    If I use rank 4, total is 7+7+7+4 + 1 = 26.
        #    Still better than 39.
        
        #    Let's try to find rank 3.
        #    C0 = A00 B0 + A01 B1
        #    C1 = A10 B0 + A11 B1
        #    M1 = (A00+A11)(B0+B1)
        #    M2 = (A10+A11)(B0)
        #    M3 = (A00+A01)(B1)
        #    C0 = M1 - M2 - M3 + (A01 B1 + A10 B0)? No.
        
        #    Let's stick to rank 4 for 2x2x1 to be safe.
        #    Or just use rank 1 for 2x1x7 and rank 7 for 2x2x2 blocks.
        
        #    Let's implement the 2x2x2 Strassen carefully.
        
        pass

    # Implementation of 2x2x2 Strassen
    def strassen_222(i_off, k_off, j_off):
        # i, k, j are local 0,1
        # Global indices:
        # u_idx = (i_off+i)*m + (k_off+k)  (m=2 for this block)
        # v_idx = (k_off+k)*p + (j_off+j) (p=2 for this block)
        # w_idx = (j_off+j)*n + (i_off+i) (n=2 for this block)
        
        # Note: m, p, n are local dimensions for the block.
        # Here n=2, m=2, p=2.
        
        def u_idx(i, k): return (i_off+i)*2 + (k_off+k)
        def v_idx(k, j): return (k_off+k)*2 + (j_off+j)
        def w_idx(j, i): return (j_off+j)*2 + (i_off+i)
        
        terms = []
        
        # 1
        terms.append(({u_idx(0,0):1, u_idx(1,1):1}, {v_idx(0,0):1, v_idx(1,1):1}, {w_idx(0,0):1, w_idx(1,1):1}))
        # 2
        terms.append(({u_idx(1,0):1, u_idx(1,1):1}, {v_idx(0,0):1}, {w_idx(1,0):1, w_idx(1,1):-1}))
        # 3
        terms.append(({u_idx(0,0):1}, {v_idx(1,1):1, v_idx(0,1):-1}, {w_idx(0,1):1, w_idx(1,1):1}))
        # 4
        terms.append(({u_idx(1,1):1}, {v_idx(1,0):1, v_idx(0,0):-1}, {w_idx(0,0):1, w_idx(1,0):1}))
        # 5
        terms.append(({u_idx(0,0):1, u_idx(0,1):1}, {v_idx(1,1):1}, {w_idx(0,0):-1, w_idx(0,1):1}))
        # 6
        terms.append(({u_idx(1,0):1, u_idx(0,0):-1}, {v_idx(0,0):1, v_idx(0,1):1}, {w_idx(1,1):1}))
        # 7
        terms.append(({u_idx(0,1):1, u_idx(1,1):-1}, {v_idx(1,0):1, v_idx(1,1):1}, {w_idx(0,0):1}))
        
        for u, v, w in terms:
            add_rank1(u, v, w)

    # Implementation of 2x2x1 (Rank 4 for safety)
    def diag_221(i_off, k_off, j_off):
        # j is fixed (0)
        # C[i, 0] = sum_k A[i, k] B[k, 0]
        # 4 terms:
        # A00 B0 -> C0
        # A01 B0 -> C0
        # A10 B0 -> C1
        # A11 B0 -> C1
        
        def u_idx(i, k): return (i_off+i)*2 + (k_off+k)
        def v_idx(k): return (k_off+k)*1 + j_off # p=1, so v_idx = k*1 + 0?
        # Wait, v_idx = k*p + j. p=1, j=0. So v_idx = k.
        # But v is global index in m*p.
        # Local m=2, p=1.
        # Global v_idx = (k_off+k)*1 + (j_off+0).
        def v_idx(k): return (k_off+k)*1 + j_off
        def w_idx(i): return (j_off+0)*2 + (i_off+i)
        # w_idx = j*n + i. n=2.
        
        terms = [
            ({u_idx(0,0):1}, {v_idx(0):1}, {w_idx(0):1}),
            ({u_idx(0,1):1}, {v_idx(1):1}, {w_idx(0):1}),
            ({u_idx(1,0):1}, {v_idx(0):1}, {w_idx(1):1}),
            ({u_idx(1,1):1}, {v_idx(1):1}, {w_idx(1):1})
        ]
        
        for u, v, w in terms:
            add_rank1(u, v, w)

    # Implementation of 2x1x7 (Rank 1)
    def diag_217(i_off, k_off, j_off):
        # k is fixed (0)
        # C[i, j] = A[i, 0] B[0, j]
        # u: all i, v: k=0, w: all j
        # u_idx = i*1 + 0 = i. (m=1)
        # v_idx = 0*7 + 0 = 0. (p=7, k=0, j=0? No, v is for B. B is 1x7. k=0, j=0..6)
        # Wait, B is m x p. m=1, p=7.
        # v_idx = k*p + j = 0*7 + j = j.
        # w_idx = j*n + i = j*2 + i.
        
        u_dict = {i_off+i: 1 for i in range(2)}
        v_dict = {k_off*7 + j: 1 for j in range(7)} # k_off=0
        w_dict = {j*2 + i_off+i: 1 for j in range(7) for i in range(2)}
        
        add_rank1(u_dict, v_dict, w_dict)

    # Construct 2x3x7
    # Split into 2x2x7 (rows 0,1) and 2x1x7 (row 2)
    # 2x2x7 split into 2x2x2 (j=0,1), 2x2x2 (j=2,3), 2x2x2 (j=4,5), 2x2x1 (j=6)
    
    # 2x2x7 part:
    # i_off=0, k_off=0
    # j blocks:
    # 1. j_off=0, p=2
    strassen_222(0, 0, 0)
    # 2. j_off=2, p=2
    strassen_222(0, 0, 2)
    # 3. j_off=4, p=2
    strassen_222(0, 0, 4)
    # 4. j_off=6, p=1
    diag_221(0, 0, 6)
    
    # 2x1x7 part:
    # i_off=2, k_off=2 (since m=3, k=2 is the third row of A? No, A is 2x3. i=0,1. k=0,1,2)
    # Wait, A is n x m. n=2, m=3.
    # So i is 0,1. k is 0,1,2.
    # My split was "2x2x7 + 2x1x7".
    # This implies splitting the k dimension (m=3) into 2+1.
    # So 2x2x7 uses k=0,1. 2x1x7 uses k=2.
    
    # So for 2x1x7:
    # i_off=0, k_off=2.
    diag_217(0, 2, 0)
    
    # Convert to numpy arrays
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
