"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135

    n, m, p = DIMENSIONS  # 4, 4, 5

    def schoolbook():
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Attempt rank-20: 4x4x4 (rank 16 schoolbook) + 4x4x1 (rank 4)
    # But schoolbook 4x4x4 is rank 16, which is just the trivial construction.
    # Let's try a better 4x4x4 decomposition.
    
    # Use Strassen-like or Winograd for 4x4x4.
    # Known: 4x4x4 can be done with rank 14 (Strassen's algorithm gives 49 for 4x4x4? No.)
    # Actually, the best known for 4x4x4 is 49 (Strassen recursive) or better.
    # But there are explicit constructions.
    
    # Let me try: split 4x4x4 into four 2x2x2 blocks + cross terms.
    # This gets complicated. Let me just use a verified construction.
    
    # Strategy: Use rank-20 with the following:
    # Terms 0-15: schoolbook for 4x4x4 (each (i,j) pair for k=0..3)
    #   Wait, schoolbook for 4x4x4 is rank 16: one term per (i,j) pair.
    #   U[i*4+j, t=i*4+j] = 1
    #   V[j*5+k, t=i*4+j] = 1 for k=0..3
    #   W[k*4+i, t=i*4+j] = 1 for k=0..3
    #   But this doesn't work because V[j*5+k] and W[k*4+i] depend on k.
    
    # Correct schoolbook for 4x4x4 (rank 16):
    # For each (i,j) in 0..3 x 0..3, term t = i*4+j:
    #   U[i*4+j, t] = 1
    #   V[j*4+k, t] = 1 for all k in 0..3  -- NO, V depends on (j,k)
    #   
    # The issue: in the full 4x4x5 problem, V has shape (20, rank) indexed by (j,k) with j in 0..3, k in 0..4.
    # For the 4x4x4 part, we only use k=0..3.
    # Schoolbook for 4x4x4: for each (i,j,k), one term. That's rank 64, not 16!
    # I was confused. Schoolbook for n×m×p is rank n*m*p.
    # So 4x4x4 schoolbook is rank 64, and 4x4x5 is rank 80.
    
    # So I need a genuinely better decomposition.
    
    # Let me try the Winograd F(4) construction properly.
    # Reference: The Winograd F(4) algorithm for 4x4 matrix multiplication uses 16 multiplications.
    # The transform matrices are 16x16 (or equivalently, 4x4 with Kronecker structure).
    
    # F(2) transform: 
    #   x0 = a0
    #   x1 = a1  
    #   x2 = a0 + a1
    #   x3 = a0 - a1
    # Inverse: a0 = (x0 + x2)/2, a1 = (x2 - x0)/2 ... but we need exact integer/half-integer.
    
    # For matrix multiplication C = A*B:
    # F(2) on A (2x2): 
    #   A' = [[a00, a01], [a10, a11]]
    #   Transform: 
    #     M_A = [[1, 0], [0, 1], [1, 1], [1, -1]]  (4x2)
    #   So U[i*2+j, t] for t=0..3, i,j in 0..1
    #     t=0: i=0,j=0: U[0,0]=1, U[1,0]=0, U[2,0]=1, U[3,0]=0 -- no, this is per element.
    
    # Let me look at this differently. For 2x2x2, the 7-term Strassen:
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
    
    # For 4x4x4, we can use a Kronecker product of two 2x2x2 Strassen algorithms.
    # This gives rank 7*7 = 49, which is worse than schoolbook 64? No, 49 < 64.
    # But we want even better.
    
    # Given time constraints, let me try to construct a rank-20 decomposition
    # using the following idea:
    # For the 4x4x5 problem, note that it's equivalent to multiplying a 4x4 matrix
    # by a 4x5 matrix. We can view this as 5 separate 4x4x1 multiplications
    # (each column of B), but with sharing.
    
    # A 4x4x1 multiplication (A * column of B) has rank 4 (schoolbook: one term per row of A).
    # But we can do better using the structure.
    
    # Actually, let me just try to find a valid rank-20 decomposition by solving
    # the system of equations, but that's 16*20*20*20 = 128000 variables and 80 equations.
    # Too large.
    
    # Let me try a different approach: use the fact that 4x4x5 = 4x4x4 + 4x4x1.
    # If I can find a rank-r decomposition for 4x4x4 and rank-s for 4x4x1,
    # the total is r+s.
    
    # For 4x4x1: A(4x4) * v(4x1) = w(4x1). This is just a matrix-vector product.
    # Rank is 4 (each output w_i = sum_j A_ij * v_j, one term per i with U[i*4+j, t]=1, V[j*5+4, t]=1, W[4*4+i, t]=1 for t=i).
    # Wait, that's rank 4: for each i in 0..3, term t=i:
    #   U[i*4+j, i] = 1 for all j
    #   V[j*5+4, i] = 1 for all j
    #   W[4*4+i, i] = 1
    # Check: sum_t U[i*4+j, t] V[j*5+4, t] W[4*4+i, t] = U[i*4+j, i] V[j*5+4, i] W[4*4+i, i] = 1*1*1 = 1 for the (i,j,4) entry. Correct!
    # And for k != 4, W[4*4+i, t] is only nonzero when k=4, so no contribution. Good.
    
    # So 4x4x1 has rank 4.
    
    # For 4x4x4, I need a good decomposition. Let me use the Strassen Kronecker product (rank 49)
    # or try to find something better.
    
    # Actually, let me just use schoolbook for 4x4x4 (rank 64) + 4x4x1 (rank 4) = rank 68.
    # That's better than 80.
    
    # Or, let me try to use a rank-16 decomposition for 4x4x4 if one exists.
    # The minimum rank for 4x4x4 is known to be at most 49 (Strassen) and at least 36 (lower bounds).
    # So rank 16 is unlikely.
    
    # Let me use the Strassen 2x2x2 (rank 7) Kronecker product for 4x4x4 (rank 49).
    # Total: 49 + 4 = 53.
    
    # Or, let me try a hybrid: use Winograd F(4) if I can get it right.
    
    # Given the complexity, let me implement a rank-68 decomposition:
    # Terms 0-63: schoolbook for 4x4x4
    # Terms 64-67: rank-4 for 4x4x1
    
    rank_total = 68
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    # Part 1: Schoolbook for 4x4x4 (terms 0-63)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j, col] = 1.0
                V[j * 5 + k, col] = 1.0
                W[k * 4 + i, col] = 1.0
                col += 1
    
    # Part 2: Rank-4 for 4x4x1 (terms 64-67)
    for i in range(4):
        t = 64 + i
        for j in range(4):
            U[i * 4 + j, t] = 1.0
            V[j * 5 + 4, t] = 1.0
        W[4 * 4 + i, t] = 1.0
    
    if verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return schoolbook()
# EVOLVE-BLOCK-END
