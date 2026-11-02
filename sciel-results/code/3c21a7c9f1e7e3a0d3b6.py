import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the tensor decomposition correctly computes matrix multiplication.
    For each (i, j, k), sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1.
    This is because the tensor T[i,j,k] = 1 for all i,j,k in the matrix multiplication
    tensor (each output C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of A[i,j]*B[j,k]
    in C[i,k] is 1).
    """
    R = len(U[0]) if len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Strategy: Block-circulant / structured decomposition
    # We partition the 12 columns into 6 groups of 2, and use Strassen on each.
    # But we can do better by exploiting the structure.
    
    # For 2x2 x 2x12, the schoolbook rank is 48.
    # Using Strassen per 2x2 block: 6 blocks * 7 = 42.
    # 
    # Can we do better? Let's think about the tensor structure.
    # The matrix multiplication tensor for (2,2,12) has:
    #   A: 2x2 (4 entries)
    #   B: 2x12 (24 entries)  
    #   C: 2x12 (24 entries)
    #
    # The tensor is T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,...,11}
    # and T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    # Actually T[i,j,k] = 1 for all valid i,j,k since C[i,k] = sum_j A[i,j] B[j,k].
    #
    # The rank of the (n,m,p) matrix multiplication tensor is known to be:
    #   For (2,2,2): rank 7 (Strassen)
    #   For (2,2,12): we can use 6 independent (2,2,2) blocks -> rank 42
    #
    # But there might be a better decomposition. Let's try to use the fact that
    # B has 24 entries and C has 24 entries, but A only has 4.
    #
    # Alternative approach: Think of this as computing 2 matrix-vector products
    # (row 0 of A times B, row 1 of A times B) where each is a 2x12 product.
    # Each row of A (2 entries) times B (2x12) gives a 1x12 vector.
    # The rank for (1,2,12) matrix multiplication is 24 (schoolbook, since it's just
    # a linear combination). Actually (1,2,p) has rank p because:
    # C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    # So we need p terms, one for each k. Rank = p = 12 for one row.
    # Two rows: rank 24? No, because the A entries are shared.
    #
    # Let me think more carefully. The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0..11}:
    # T[i,j,k] = 1 for all i,j,k.
    #
    # This is actually the tensor of the form: for each k, we have a 2x2x1 sub-tensor
    # where all entries are 1. That's the (2,2,1) multiplication tensor repeated 12 times
    # with the k-index being free.
    #
    # The (2,2,1) tensor: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #                       C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is just two independent dot products of length 2. Rank = 2 (schoolbook is optimal
    # for (2,2,1) since it's just two independent multiplications... wait no).
    #
    # Actually for (2,2,1): C[i,0] = sum_j A[i,j] B[j,0]
    # The tensor has T[i,j,0] = 1 for all i,j. This is a 2x2x1 tensor with all ones.
    # Its rank is 2 (you can write it as:
    #   term 1: A[0,0]*B[0,0]*C[0,0] + A[1,0]*B[0,0]*C[1,0] -> U=[1,0,1,0], V=[1,0], W=[1,1]
    #   term 2: A[0,1]*B[1,0]*C[0,0] + A[1,1]*B[1,0]*C[1,0] -> U=[0,1,0,1], V=[0,1], W=[1,1]
    # So rank 2 for (2,2,1).
    #
    # For (2,2,12), we have 12 independent (2,2,1) tensors (one for each k), so rank = 12*2 = 24?
    # Wait, that can't be right because schoolbook gives 48.
    #
    # Let me re-examine. The schoolbook for (2,2,12) has 2*2*12 = 48 terms.
    # Each term is A[i,j] * B[j,k] * C[i,k].
    #
    # If I group by k: for fixed k, I need to compute C[0,k] and C[1,k] from A and B[:,k].
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # For fixed k, this is a (2,2,1) problem with rank 2. So total rank = 12 * 2 = 24?
    # But wait, the A entries are shared across all k! So we can't just do 12 independent rank-2 decompositions.
    #
    # Let me think again. The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..11}.
    # We want to write T as sum_r U_r ⊗ V_r ⊗ W_r where:
    #   U_r is a vector in R^4 (indexed by (i,j))
    #   V_r is a vector in R^24 (indexed by (j,k))
    #   W_r is a vector in R^24 (indexed by (k,i))
    #
    # And T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    #
    # For this to equal 1 for all i,j,k, we need:
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    #
    # One simple construction: For each (i,j), create one term:
    #   U_r has 1 at (i,j), 0 elsewhere
    #   V_r has 1 at (j,k) for all k, 0 elsewhere
    #   W_r has 1 at (k,i) for all k, 0 elsewhere
    # But this gives 4 terms, each with V and W having 12 non-zero entries.
    # Wait, that's not how tensor rank works. Each term is a rank-1 tensor.
    #
    # Let me try: For each i in {0,1} and j in {0,1}:
    #   Term (i,j): U has 1 at position (i,j). V has 1 at positions (j,0),(j,1),...,(j,11). 
    #                W has 1 at positions (0,i),(1,i),...,(11,i).
    #   Then sum over (i,j) of U[i,j]*V[j,k]*W[k,i] = sum_{i,j} delta(i,j)*[j=j]*[k=i] 
    #   Hmm, this doesn't work cleanly.
    #
    # Let me just go with the Strassen-based rank-42 decomposition which is known to work.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen U vectors (A factors)
    # A indices: 0: A00, 1: A01, 2: A10, 3: A11
    u_strassen = [
        [1.0, 0.0, 0.0, 1.0],  # M1
        [0.0, 0.0, 1.0, 1.0],  # M2
        [1.0, 0.0, 0.0, 0.0],  # M3
        [0.0, 0.0, 0.0, 1.0],  # M4
        [1.0, 1.0, 0.0, 0.0],  # M5
        [-1.0, 0.0, 1.0, 0.0], # M6
        [0.0, 1.0, 0.0, -1.0]  # M7
    ]
    
    for l in range(6):
        # V vectors for block l
        # B indices:
        # B_0,2l -> 2l
        # B_0,2l+1 -> 2l+1
        # B_1,2l -> 12+2l
        # B_1,2l+1 -> 12+2l+1
        
        b00_idx = 2*l
        b01_idx = 2*l + 1
        b10_idx = 12 + 2*l
        b11_idx = 12 + 2*l + 1
        
        v_strassen = [
            {b00_idx: 1.0, b11_idx: 1.0},   # M1: B00 + B11
            {b00_idx: 1.0},                  # M2: B00
            {b01_idx: 1.0, b11_idx: -1.0},   # M3: B01 - B11
            {b10_idx: 1.0, b00_idx: -1.0},   # M4: B10 - B00
            {b11_idx: 1.0},                  # M5: B11
            {b00_idx: 1.0, b01_idx: 1.0},    # M6: B00 + B01
            {b10_idx: 1.0, b11_idx: 1.0}     # M7: B10 + B11
        ]
        
        # W vectors for block l
        # C indices (W index = k*n + i = 2k + i):
        # C_0,2l -> 4l
        # C_1,2l -> 4l+1
        # C_0,2l+1 -> 4l+2
        # C_1,2l+1 -> 4l+3
        
        c00_idx = 4*l
        c10_idx = 4*l + 1
        c01_idx = 4*l + 2
        c11_idx = 4*l + 3
        
        # Reconstruction:
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        
        w_strassen = [
            {c00_idx: 1.0, c11_idx: 1.0},    # M1
            {c10_idx: 1.0},                  # M2
            {c01_idx: 1.0, c11_idx: -1.0},   # M3
            {c00_idx: -1.0, c10_idx: 1.0},   # M4
            {c01_idx: 1.0, c11_idx: -1.0},   # M5
            {c00_idx: -1.0, c11_idx: 1.0},   # M6
            {c00_idx: 1.0}                   # M7
        ]
        
        for m_idx in range(7):
            r = l * 7 + m_idx
            
            # Fill U
            for i in range(4):
                U[i][r] = u_strassen[m_idx][i]
                
            # Fill V
            for idx, val in v_strassen[m_idx].items():
                V[idx][r] += val
                
            # Fill W
            for idx, val in w_strassen[m_idx].items():
                W[idx][r] += val
                
    # Verify
    if _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    fb = _schoolbook(n, m, p)
    return fb

# EVOLVE-BLOCK-END
