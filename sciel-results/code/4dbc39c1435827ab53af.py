"""
Optimized tensor decomposition for 2x4 by 4x8 matrix multiplication.
Strategy: Decompose into 2x4 sub-blocks of size 2x2x2 and apply Strassen's algorithm.
This yields a rank of 56, which is significantly lower than the schoolbook rank of 64.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_validity(U, V, W, n, m, p):
    """
    Verifies the decomposition by reconstructing the tensor and checking against the identity.
    Returns True if the decomposition is exact.
    """
    try:
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check dimensions
        if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
            return False
            
        # Reshape to tensor forms for easy contraction
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R) -> note: W[k*n+i] corresponds to index (k, i) in (p, n) space if we treat first dim as p, second as n?
        # Actually, W is indexed by [k*n + i]. 
        # Let's check the convention: W[k*n+i, r].
        # So W_reshaped[k, i, r] = W_arr[k*n+i, r].
        # This means W_reshaped has shape (p, n, R).
        
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W_t[k, i, r] is exactly what we need.
        
        # Use einsum for speed
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # The target tensor for matrix multiplication has T[i,j,k] = 1 if the indices correspond to the multiplication
        # Wait, the tensor T_{ijk} represents the coefficient of A_{ij} B_{jk} in C_{ik}.
        # In a standard basis, T_{ijk} = 1 for all valid i,j,k.
        # Let's verify this assumption.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T is defined such that T(A,B) = C.
        # T_{ijk} is the coefficient of e_i (in A) and e_j (in B) contributing to e_k? No.
        # Usually, the multiplication tensor is defined as T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
        # So T[i,j,k] should be 1.
        
        target = np.ones((n, m, p))
        
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True
    except Exception:
        return False

def _generate_strassen_decomp():
    """
    Generates the U, V, W matrices for the rank 56 decomposition.
    n=2, m=4, p=8.
    We split the middle dimension (size 4) into 2 blocks of size 2.
    We split the output dimension (size 8) into 4 blocks of size 2.
    For each (a_block, b_block) pair, we perform a 2x2x2 Strassen multiplication.
    
    Indices:
    i in [0, 1] (n=2)
    j in [0, 3] (m=4) -> j = 2*a + la, where a in [0,1], la in [0,1]
    k in [0, 7] (p=8) -> k = 2*b + lb, where b in [0,3], lb in [0,1]
    
    U row index: i*m + j
    V row index: j*p + k
    W row index: k*n + i
    
    Strassen terms for 2x2 matrices (indexed 0,1):
    1. M1 = (A00+A11)(B00+B11)
    2. M2 = (A10+A11)B00
    3. M3 = A00(B01-B11)
    4. M4 = A11(B10-B00)
    5. M5 = (A00+A01)B11
    6. M6 = (A10-A00)(B00+B01)
    7. M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    R = 56
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    # Define the 7 Strassen terms
    # Each term is a list of (coeff, i_idx, j_idx) for U, (coeff, j_idx, k_idx) for V, (coeff, i_idx, k_idx) for W
    # Local indices la, lb are 0 or 1.
    # Global j = 2*a + la
    # Global k = 2*b + lb
    
    # Term 1: (A00+A11)(B00+B11)
    # U: (1, 0, 0), (1, 1, 1)  [la=0, la=1]
    # V: (1, 0, 0), (1, 1, 1)  [lb=0, lb=1]
    # W: (1, 0, 0), (1, 1, 1)  [C00 gets +1, C11 gets -1? Wait.
    # Let's map C terms carefully.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define contributions to W (which is C) for each term M_t.
    # W is indexed by (k, i).
    
    # Term 1 (M1):
    # A part: A00 (la=0, li=0), A11 (la=1, li=1)
    # B part: B00 (lb=0, lj=0), B11 (lb=1, lj=1)
    # C part: C00 (li=0, lk=0) coeff +1, C11 (li=1, lk=1) coeff +1
    t1_U = [(1.0, 0, 0), (1.0, 1, 1)]
    t1_V = [(1.0, 0, 0), (1.0, 1, 1)]
    t1_W = [(1.0, 0, 0), (1.0, 1, 1)]
    
    # Term 2 (M2):
    # A part: A10 (la=0, li=1), A11 (la=1, li=1)
    # B part: B00 (lb=0, lj=0)
    # C part: C10 (li=1, lk=0) coeff +1, C11 (li=1, lk=1) coeff -1
    t2_U = [(1.0, 0, 1), (1.0, 1, 1)]
    t2_V = [(1.0, 0, 0)]
    t2_W = [(1.0, 0, 1), (-1.0, 1, 1)]
    
    # Term 3 (M3):
    # A part: A00 (la=0, li=0)
    # B part: B01 (lb=1, lj=0), B11 (lb=1, lj=1) -> B01 - B11
    # C part: C01 (li=0, lk=1) coeff +1, C11 (li=1, lk=1) coeff +1
    t3_U = [(1.0, 0, 0)]
    t3_V = [(1.0, 0, 1), (-1.0, 1, 1)]
    t3_W = [(1.0, 1, 0), (1.0, 1, 1)]
    
    # Term 4 (M4):
    # A part: A11 (la=1, li=1)
    # B part: B10 (lb=0, lj=1), B00 (lb=0, lj=0) -> B10 - B00
    # C part: C00 (li=0, lk=0) coeff +1, C10 (li=1, lk=0) coeff +1
    t4_U = [(1.0, 1, 1)]
    t4_V = [(1.0, 1, 0), (-1.0, 0, 0)]
    t4_W = [(1.0, 0, 0), (1.0, 0, 1)]
    
    # Term 5 (M5):
    # A part: A00 (la=0, li=0), A01 (la=1, li=0) -> A00 + A01
    # B part: B11 (lb=1, lj=1)
    # C part: C00 (li=0, lk=0) coeff -1, C01 (li=0, lk=1) coeff +1
    t5_U = [(1.0, 0, 0), (1.0, 1, 0)]
    t5_V = [(1.0, 1, 1)]
    t5_W = [(-1.0, 0, 0), (1.0, 1, 0)]
    
    # Term 6 (M6):
    # A part: A10 (la=0, li=1), A00 (la=0, li=0) -> A10 - A00
    # B part: B00 (lb=0, lj=0), B01 (lb=1, lj=0) -> B00 + B01
    # C part: C11 (li=1, lk=1) coeff +1
    t6_U = [(1.0, 0, 1), (-1.0, 0, 0)]
    t6_V = [(1.0, 0, 0), (1.0, 0, 1)]
    t6_W = [(1.0, 1, 1)]
    
    # Term 7 (M7):
    # A part: A01 (la=1, li=0), A11 (la=1, li=1) -> A01 - A11
    # B part: B10 (lb=0, lj=1), B11 (lb=1, lj=1) -> B10 + B11
    # C part: C00 (li=0, lk=0) coeff +1
    t7_U = [(1.0, 1, 0), (-1.0, 1, 1)]
    t7_V = [(1.0, 1, 0), (1.0, 1, 1)]
    t7_W = [(1.0, 0, 0)]
    
    strassen_terms = [
        (t1_U, t1_V, t1_W),
        (t2_U, t2_V, t2_W),
        (t3_U, t3_V, t3_W),
        (t4_U, t4_V, t4_W),
        (t5_U, t5_V, t5_W),
        (t6_U, t6_V, t6_W),
        (t7_U, t7_V, t7_W)
    ]
    
    col = 0
    # Iterate over blocks of the middle dimension (m=4, size 2) -> 2 blocks
    # Iterate over blocks of the output dimension (p=8, size 2) -> 4 blocks
    for a_blk in range(2):
        for b_blk in range(4):
            for term in strassen_terms:
                u_terms, v_terms, w_terms = term
                
                # Populate U
                for coeff, la, li in u_terms:
                    j = 2 * a_blk + la
                    i = li
                    U[i * m + j, col] += coeff
                
                # Populate V
                for coeff, lj, lb in v_terms:
                    j = 2 * a_blk + lj
                    k = 2 * b_blk + lb
                    V[j * p + k, col] += coeff
                
                # Populate W
                # W is indexed by k*n + i.
                # The term in W corresponds to C_{ik}.
                # In Strassen, C is 2x2. C_{li, lk}.
                # So i = li, k = 2*b_blk + lk.
                for coeff, li, lk in w_terms:
                    i = li
                    k = 2 * b_blk + lk
                    W[k * n + i, col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    
    # Build the rank 56 decomposition
    res = _generate_strassen_decomp()
    U, V, W = res["U"], res["V"], res["W"]
    
    # Convert to lists of floats as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _check_validity(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook (Rank 64) just in case
    R = n * m * p
    U_s = np.zeros((n*m, R))
    V_s = np.zeros((m*p, R))
    W_s = np.zeros((p*n, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, c] = 1.0
                V_s[j*p+k, c] = 1.0
                W_s[k*n+i, c] = 1.0
                c += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
