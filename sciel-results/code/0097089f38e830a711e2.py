"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2_decomp():
    """
    Return the rank-7 decomposition of 2x2 * 2x2 matrix multiplication.
    U, V, W are lists of 7 tuples (or a 7x4 array) such that:
    C = sum_t U_t * A * V_t
    where U_t, V_t are 2x2 matrices.
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each product t, we need U_t and V_t such that U_t @ A @ V_t gives the product.
    # Actually, Strassen's formula is: M_t = (U_t @ A) * (V_t @ B) or similar.
    # Let's use the standard form: M_t = (linear combo of A) * (linear combo of B)
    # We can represent this as: M_t = (U_t @ A) @ (V_t @ B) is not quite right.
    # The standard Strassen is: M_t = P_t(A) * Q_t(B) where P_t and Q_t are linear.
    # For tensor decomposition: T(A,B) = sum_t P_t(A) * Q_t(B)
    # In our case, C = A * B, so we want C_ij = sum_t P_t(A) * Q_t(B)
    # But actually, the tensor decomposition for matrix multiplication is:
    # C = sum_t (U_t @ A) @ (V_t @ B) is not standard.
    # 
    # The standard form is: (A*B)_ij = sum_k A_ik * B_kj
    # Strassen rewrites this as: (A*B)_ij = sum_t M_t * w_t_ij
    # where M_t = P_t(A) * Q_t(B) and P_t, Q_t are linear forms.
    # 
    # For the tensor decomposition interface, we need:
    # C = sum_t U_t[:,i,j] * A_i * V_t[:,j,k] ... 
    # 
    # Let me think about this differently. The multiplication tensor for 2x2*2x2 has rank 7.
    # The decomposition is: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^4 (flattened A), v_t is in R^4 (flattened B), w_t is in R^4 (flattened C).
    #
    # Strassen's 7 terms:
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   u1 = [1,0,0,1] (A00 + A11)
    #   v1 = [1,0,0,1] (B00 + B11)
    #   w1 = [1,0,0,-1] for C00, [0,0,0,0] for C01, [0,0,0,0] for C10, [1,0,0,1] for C11
    #   Wait, w1 should be the coefficient vector for C.
    #   C00 += M1, C11 += M1, so w1 = [1, 0, 0, 1]
    #
    # Term 2: M2 = (A10+A11)(B00)
    #   u2 = [0,0,1,1]
    #   v2 = [1,0,0,0]
    #   C10 += M2, C11 -= M2, so w2 = [0, 0, 1, -1]
    #
    # Term 3: M3 = A00(B01-B11)
    #   u3 = [1,0,0,0]
    #   v3 = [0,1,0,-1]
    #   C01 += M3, C11 += M3, so w3 = [0, 1, 0, 1]
    #
    # Term 4: M4 = A11(B10-B00)
    #   u4 = [0,0,0,1]
    #   v4 = [0,0,1,-1]
    #   C00 += M4, C10 += M4, so w4 = [1, 0, 1, 0]
    #
    # Term 5: M5 = (A00+A01)(B11)
    #   u5 = [1,1,0,0]
    #   v5 = [0,0,0,1]
    #   C00 -= M5, C01 += M5, so w5 = [-1, 1, 0, 0]
    #
    # Term 6: M6 = (A10-A00)(B00+B01)
    #   u6 = [-1,0,1,0]
    #   v6 = [1,1,0,0]
    #   C11 += M6, so w6 = [0, 0, 0, 1]
    #
    # Term 7: M7 = (A01-A11)(B10+B11)
    #   u7 = [0,1,0,-1]
    #   v7 = [0,0,1,1]
    #   C00 += M7, so w7 = [1, 0, 0, 0]
    
    u = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ], dtype=float)
    
    v = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ], dtype=float)
    
    w = np.array([
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, -C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> -C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0],  # M7 -> C00
    ], dtype=float)
    
    return u, v, w

def _build_decomposition_2x2x16():
    """
    Build a decomposition for 2x2 * 2x16 multiplication.
    
    We view A as 2x2 and B as 2x16. We partition B into 8 blocks of 2 columns each.
    For each block, we use the rank-7 Strassen decomposition for 2x2 * 2x2.
    Total rank = 8 * 7 = 56.
    
    The output C is 2x16.
    
    Interface: U (n*m=4, rank), V (m*p=32, rank), W (p*n=32, rank)
    - U[i*m+j, c] corresponds to A[i,j] coefficient
    - V[j*p+k, c] corresponds to B[j,k] coefficient  
    - W[k*n+i, c] corresponds to C[k,i] coefficient (trace convention: W[k*n+i, r])
    
    Wait, the trace convention is W[k*n+i, r]. Let me re-read:
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no, that's not right either.
    
    Actually, for matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition is: C = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    
    The flattening convention:
    - U is indexed by (i, j) -> row i*m + j
    - V is indexed by (j, k) -> row j*p + k
    - W is indexed by (k, i) -> row k*n + i (this is the "trace convention")
    
    So the decomposition satisfies:
    C[i,k] = sum_j A[i,j] * B[j,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    
    Hmm, that's not quite right. Let me think again.
    
    The standard tensor decomposition for matrix multiplication:
    T(A,B) = C where C_ij = sum_k A_ik B_kj
    
    This can be written as: C = sum_r (U_r @ A) @ (V_r @ B) ... no.
    
    Actually, the correct form is:
    C_{ik} = sum_j A_{ij} B_{jk}
    
    We want: C_{ik} = sum_r w_{ik}^{(r)} * u_{ij}^{(r)} * v_{jk}^{(r)} summed appropriately.
    
    More precisely: C_{ik} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} * V_{j*p+k, r})
    
    No, that's still not right. Let me be very explicit.
    
    The tensor T has indices (i, j, k) and T_{ijk} = 1 if the multiplication uses A_{ij} and B_{jk} to produce C_{ik}.
    Actually, T_{ijk} = delta_{ik} in some sense... no.
    
    For matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    
    The tensor is: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (coefficient)
    where the coefficient is 1 for each (i,j,k) triple.
    
    So T_{ijk} = 1 for all i,j,k (in the appropriate ranges).
    
    A rank-r decomposition is: T_{ijk} = sum_{r=1}^R U_{ir} V_{jr} W_{kr}
    
    Wait, but the indices are: i for A's row, j for A's col / B's row, k for B's col / C's col.
    And C's row is i, C's col is k.
    
    So: C_{ik} = sum_j A_{ij} B_{jk} = sum_{j,r} A_{ij} B_{jk} * (something)
    
    The correct decomposition: C_{ik} = sum_r (sum_j U_{i*m+j, r} * A_{ij}) * (sum_k' ... ) 
    
    I think the correct form is:
    C_{ik} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} * A_{ij}) * (sum_k' V_{j*p+k', r} * B_{j k'})
    
    No, that doesn't work dimensionally.
    
    Let me look at the seed program:
    U[i*m+j, column] = 1
    V[j*p+k, column] = 1
    W[k*n+i, column] = 1
    
    So for each (i,j,k), we have a rank-1 term with U having 1 at position (i*m+j), V having 1 at (j*p+k), W having 1 at (k*n+i).
    
    The reconstruction would be:
    C_{ik} = sum_{i',j',k',r} ... 
    
    Actually, I think the convention is:
    C_{ik} = sum_r (sum_j U_{i*m+j, r} * A_{ij}) * (sum_k' V_{k'*m+j, r} * B_{k' j}) ... 
    
    This is getting confusing. Let me just verify with the seed program that it works, and then adapt.
    
    The key insight from the seed: for each (i,j,k) triple, we create a rank-1 term. The total rank is n*m*p.
    
    For our case: n=2, m=2, p=16, so rank = 64.
    
    To reduce rank, we can group the p=16 columns into blocks and use Strassen within each block.
    
    Let me partition B (2x16) into 8 blocks of 2 columns: B[:, 2b:2b+2] for b=0,...,7.
    
    For each block, we compute C[:, 2b:2b+2] = A * B[:, 2b:2b+2] using Strassen (rank 7).
    
    Total rank = 8 * 7 = 56.
    
    Now, for each block b and each Strassen term t (0..6):
    - The Strassen term computes M_{b,t} = P_t(A) * Q_t(B[:, 2b:2b+2])
    - This contributes to C[:, 2b:2b+2] via the w_t coefficients.
    
    Let me build this explicitly.
    """
    n, m, p = 2, 2, 16
    u_strassen, v_strassen, w_strassen = _strassen_2x2_decomp()
    # u_strassen: 7x4, v_strassen: 7x4, w_strassen: 7x4
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7  # 56
    
    U = np.zeros((n * m, rank))  # 4 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 32 x 56
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            
            # U part: coefficients for A
            # u_strassen[t] is a 4-vector for A flattened as (A00, A01, A10, A11)
            # U index: i*m + j
            u_vec = u_strassen[t]  # length 4
            for idx in range(4):
                U[idx, col] = u_vec[idx]
            
            # V part: coefficients for B block b
            # v_strassen[t] is a 4-vector for B block flattened as (B00, B01, B10, B11)
            # B block b has columns 2b and 2b+1
            # V index: j*p + k
            v_vec = v_strassen[t]  # length 4
            # v_vec[0] -> B[0, 2b], v_vec[1] -> B[0, 2b+1], v_vec[2] -> B[1, 2b], v_vec[3] -> B[1, 2b+1]
            V[0 * p + (2 * b), col] = v_vec[0]
            V[0 * p + (2 * b + 1), col] = v_vec[1]
            V[1 * p + (2 * b), col] = v_vec[2]
            V[1 * p + (2 * b + 1), col] = v_vec[3]
            
            # W part: coefficients for C block b
            # w_strassen[t] is a 4-vector for C block flattened as (C00, C01, C10, C11)
            # W index: k*n + i (trace convention)
            w_vec = w_strassen[t]  # length 4
            # w_vec[0] -> C[0, 2b], w_vec[1] -> C[0, 2b+1], w_vec[2] -> C[1, 2b], w_vec[3] -> C[1, 2b+1]
            # W index for C[i, k] is k*n + i
            W[(2 * b) * n + 0, col] = w_vec[0]      # C[0, 2b]
            W[(2 * b + 1) * n + 0, col] = w_vec[1]   # C[0, 2b+1]
            W[(2 * b) * n + 1, col] = w_vec[2]       # C[1, 2b]
            W[(2 * b + 1) * n + 1, col] = w_vec[3]   # C[1, 2b+1]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    
    # Build the rank-56 decomposition using Strassen
    U, V, W = _build_decomposition_2x2x16()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Shouldn't happen, but safety check
        pass
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
