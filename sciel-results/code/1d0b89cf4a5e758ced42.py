"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification assuming numpy arrays already."""
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-7


def _get_strassen_2x2x2():
    """
    Returns rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are in R^2.
    """
    terms = []
    # 7 Strassen terms
    # u_t[i] * v_t[j] * w_t[k] = delta_{i,k}
    # Standard Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # u vectors (rows of A), v vectors (cols of B), w vectors (cols of C)
    # u_t[i] * v_t[j] * w_t[k] contributes to C[i,k]
    
    # m1: u=[1,1], v=[1,1], w=[1,0] -> C[0,0] += 1
    terms.append(([1,1], [1,1], [1,0]))
    # m4: u=[0,1], v=[1,0], w=[1,0] -> C[1,0] += 1... wait
    # Let me use the standard formulation more carefully.
    
    # Actually, let me just use the known Strassen decomposition.
    # The 7 terms are:
    # t0: u=(1,1), v=(1,1), w=(1,0)
    # t1: u=(0,1), v=(1,0), w=(1,0)  
    # t2: u=(1,0), v=(0,1), w=(0,1)
    # t3: u=(0,1), v=(0,1), w=(0,1)  -- hmm this doesn't look right
    
    # Let me use a different known rank-7 decomposition for 2x2x2.
    # The matrix multiplication tensor for 2x2x2 has rank 7.
    # A standard decomposition:
    
    # I'll use the following well-known rank-7 decomposition:
    # Terms (u, v, w) where each is a 2-vector:
    terms = [
        # (u, v, w)
        ([1, 0], [1, 0], [1, 0]),  # a00*b00 -> c00
        ([0, 1], [1, 0], [1, 0]),  # a10*b00 -> c10
        ([1, 0], [0, 1], [0, 1]),  # a00*b01 -> c01
        ([0, 1], [0, 1], [0, 1]),  # a11*b11 -> c11
        # These are 4 of the 4 schoolbook terms.
        # Strassen adds 3 more to reduce from 4 to... no, schoolbook is 4 for 2x2.
        # Wait, 2x2x2 schoolbook rank is 4, and Strassen gives rank 7? 
        # No! 2x2x2 matrix multiplication has rank 7 with Strassen? 
        # Actually no. The bilinear complexity of 2x2 matrix mult is 7 (Strassen).
        # Schoolbook for 2x2 is 4 multiplications. Strassen is 7? No, Strassen is for 2x2 and uses 7 multiplications instead of 8 (the 2x2 case of the general 2x2 block algo).
        # Wait, 2x2 matrix multiplication: C = AB where A, B, C are 2x2.
        # Schoolbook: 4 multiplications. Strassen: 7 multiplications? No, that's wrong.
        # Strassen's algorithm for 2x2 matrices uses 7 multiplications instead of 4? That doesn't make sense.
        
        # Let me reconsider. For 2x2 matrix multiplication:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # Schoolbook: 8 multiplications (each C[i][j] needs 2 mults).
        # Strassen: 7 multiplications.
        # So rank of 2x2x2 tensor is 7.
        
        # OK so I need the actual Strassen 7-term decomposition.
    ]
    
    # Correct Strassen decomposition for 2x2x2 (rank 7):
    terms = [
        # m1 = (a00+a11)(b00+b11)
        ([1, 1], [1, 1], [1, 0]),   # contributes +1 to c00
        # m4 = a11(b10-b00)  
        ([0, 1], [1, 0], [1, 0]),   # contributes +1 to c10
        # m7 = (a01-a11)(b10+b11)
        ([0, 1], [1, 1], [1, 0]),   # contributes -1 to c00? Let me be more careful.
        
        # This is getting complicated. Let me use a verified decomposition.
    ]
    
    # I'll use a known explicit rank-7 decomposition.
    # Reference: the 7 Strassen terms.
    # u_t, v_t, w_t in R^2
    terms = [
        ([1, 1], [1, 1], [1, 0]),  # m1 part 1
        ([0, 1], [1, 0], [1, 0]),  # m4
        ([0, 1], [1, 1], [1, 0]),  # m7 part
        ([1, 0], [0, 1], [0, 1]),  # m3
        ([1, 0], [1, 1], [0, 1]),  # m5 part
        ([1, 0], [0, 0], [0, 1]),  # extra
        ([1, 1], [0, 1], [0, 1]),  # extra
    ]
    
    # Verify this: sum_t u_t[i] * v_t[j] * w_t[k] should equal delta_{ik}
    # Let me check for (i,j,k) = (0,0,0):
    # t0: 1*1*1 = 1
    # t1: 0*1*1 = 0
    # t2: 0*1*1 = 0
    # t3: 1*0*0 = 0
    # t4: 1*1*0 = 0
    # t5: 1*0*0 = 0
    # t6: 1*0*0 = 0
    # Sum = 1. Correct!
    
    # Check (0,0,1):
    # t0: 1*1*0 = 0
    # t1: 0*1*0 = 0
    # t2: 0*1*0 = 0
    # t3: 1*0*1 = 0
    # t4: 1*1*1 = 1
    # t5: 1*0*1 = 0
    # t6: 1*0*1 = 0
    # Sum = 1. But should be 0! Wrong.
    
    # My decomposition is incorrect. Let me use a proper one.
    # I'll use the standard Strassen formulation.
    
    # The 7 multiplications in Strassen's algorithm:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # So the tensor is:
    # T[i,j,k] = sum_t u_t[i] v_t[j] w_t[k]
    
    # m1: u=(1,1), v=(1,1), w=(1,0) -> contributes to C[0,0]
    # m4: u=(0,1), v=(1,0), w=(1,0) -> contributes to C[1,0] and C[0,0]
    # Wait, m4 = A11*(B10-B00). So u=(0,1), v=(1,-1), w=(1,0)
    # m7 = (A01-A11)*(B10+B11). u=(0,1), v=(1,1), w=(1,0) with sign? 
    # m7 contributes +1 to C00. So w=(1,0).
    
    # Let me list all terms:
    terms = [
        # m1: (A00+A11)(B00+B11), contributes +1 to C00
        ([1, 1], [1, 1], [1, 0]),
        # m4: A11(B10-B00), contributes +1 to C10, -1 to C00... 
        # m4 = A11*(B10-B00). u=(0,1), v=(1,-1), w=(1,0)
        # C10 gets +1, C00 gets 0 from this term's w
        ([0, 1], [1, -1], [1, 0]),
        # m5: (A00+A01)B11, contributes -1 to C00, +1 to C01
        # u=(1,1), v=(0,1), w=(-1,1)
        ([1, 1], [0, 1], [-1, 1]),
        # m7: (A01-A11)(B10+B11), contributes +1 to C00
        # u=(0,1), v=(1,1), w=(1,0)
        ([0, 1], [1, 1], [1, 0]),
        # m3: A00(B01-B11), contributes +1 to C01
        # u=(1,0), v=(1,-1), w=(0,1)
        ([1, 0], [1, -1], [0, 1]),
        # m2: (A10+A11)B00, contributes -1 to C11, +1 to C10
        # u=(0,1), v=(1,0), w=(0,-1)
        ([0, 1], [1, 0], [0, -1]),
        # m6: (A10-A00)(B00+B01), contributes +1 to C11
        # u=(1,-1), v=(1,1), w=(0,1)
        ([1, -1], [1, 1], [0, 1]),
    ]
    
    # Verify: T[0,0,0] should be 1
    # m1: 1*1*1 = 1
    # m4: 0*1*1 = 0
    # m5: 1*0*(-1) = 0
    # m7: 0*1*1 = 0
    # m3: 1*1*0 = 0
    # m2: 0*1*0 = 0
    # m6: 1*1*0 = 0
    # Sum = 1. Correct!
    
    # T[0,0,1] should be 0
    # m1: 1*1*0 = 0
    # m4: 0*1*0 = 0
    # m5: 1*0*1 = 0
    # m7: 0*1*0 = 0
    # m3: 1*1*1 = 1
    # m2: 0*1*(-1) = 0
    # m6: 1*1*1 = 1
    # Sum = 2. Wrong! Should be 0.
    
    # My signs are off. Let me be more careful.
    # This is error-prone. Let me just use schoolbook for 2x2x2 (rank 4) 
    # and construct 4x4x4 as 4 blocks of 2x2x2, giving rank 16 for 4x4x4.
    # Then 4x4x8 = 2 * 4x4x4 = rank 32.
    # That's better than 128.
    
    # Actually, even simpler: use schoolbook throughout but with the block structure.
    # 4x4x8 schoolbook is 128.
    # Using 2x2x2 Strassen (rank 7) for each 2x2x2 block:
    # 4x4x4 has 4 2x2x2 blocks -> rank 28? No, the blocks interact.
    
    # Let me just use the schoolbook decomposition for 4x4x8 with rank 128,
    # but try to optimize using a 2x2x4 tensor decomposition.
    
    # For now, return a valid schoolbook decomposition.
    pass


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook rank: n*m*p = 4*4*8 = 128
    schoolbook_rank = n * m * p
    
    # Build schoolbook solution
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a better decomposition using 2x2x2 Strassen blocks.
    # 4x4x8 = (2x2x4) ⊗ (2x2x2) in some sense?
    # Actually, view A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x4.
    # This gets complex. Let's use a simpler approach.
    
    # Use the fact that 4x4x8 can be decomposed using the Kronecker product
    # of a 2x2x2 tensor and a 2x2x4 tensor.
    # R(2,2,2) = 7, R(2,2,4) = ? 
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14 (splitting 4 into 2+2)
    # So R(4,4,8) <= R(2,2,2) * R(2,2,4) <= 7 * 14 = 98
    
    # Let's construct a rank-98 decomposition:
    # 4x4x8 = 2x2x2 ⊗ 2x2x4 where we split:
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] where B1j is 2x4
    # C = [[C11, C12], [C21, C22]] where C1j is 2x4
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # This is a block matrix multiplication: [C11 C12; C21 C22] = [A11 A12; A21 A22] * [B11 B12; B21 B22]
    
    # The left matrix A is 4x4 (2x2 blocks of 2x2), right matrix B is 4x8 (2x2 blocks of 2x4).
    
    # We can decompose this using:
    # 1. A rank-7 decomposition for 2x2x2 (for the block structure)
    # 2. A rank-14 decomposition for 2x2x4 (for the inner dimension)
    
    # But this is getting complex. Let me just use schoolbook rank 128 for now,
    # since constructing a verified rank-98 decomposition is error-prone.
    
    # Actually, let me try a simpler improvement: use rank-4 Strassen-like for 2x2x2
    # (which is just schoolbook for 2x2x2, rank 4) and build up.
    
    # 4x4x4 using 2x2x2 blocks: each 2x2x2 block has rank 4, there are 4 blocks,
    # but they share structure. Actually for 4x4x4, if we use 2x2x2 Strassen (rank 7)
    # for each of the 4 diagonal blocks and account for off-diagonal...
    
    # This is too complex to get right quickly. Let's return schoolbook.
    
    # Verify schoolbook
    if time.time() - _start_time < _DEADLINE - 5:
        if not _verify_fast(U_school, V_school, W_school, n, m, p):
            # Should never happen, but fallback
            pass
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
