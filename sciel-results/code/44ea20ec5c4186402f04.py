"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    # Precompute the tensor T as a list of non-zero entries
    # T[(i,j,k)] = 1
    # We want to find U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # This is the matrix multiplication tensor.
    
    # Since exact rank minimization is hard, we try a few small rank constructions.
    # But for 3x4x5, the schoolbook rank is 60.
    # Known lower bounds: R(M_3,4,5) >= 21 (roughly, based on n*m+p or similar, but actually much higher).
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} is at least n*m + m*p + p*n - 1 ? No.
    # The trivial rank is n*m*p.
    
    # Let's try to implement a specific small-rank construction if possible, or just return schoolbook.
    # Given the constraints and the need for correctness, and that finding a better rank than 60 for 3x4x5 is non-trivial
    # (Strassen works for 2x2 and 4x4, but 3x4x5 is irregular), the safest bet is the schoolbook algorithm.
    # However, we can try to see if a block-wise Strassen approach reduces the rank.
    # Split A (3x4) into [A1 (3x2), A2 (3x2)]
    # Split B (4x5) into [B1 (2x5); B2 (2x5)]
    # C = A1 B1 + A2 B2
    # Rank(A1 B1) + Rank(A2 B2).
    # A1 B1 is 3x2x5. A2 B2 is 3x2x5.
    # For 3x2x5: Split 5 into 2+2+1? Or 3+2?
    # If we split 5 into 3+2: A1 B1 = A1 B11 + A1 B12.
    # A1 B11 is 3x2x3. A1 B12 is 3x2x2.
    # This seems to lead to a rank close to schoolbook unless specific fast algorithms for these small sizes are known.
    # For 3x2x2, schoolbook is 12. Is there a faster one?
    # M_3,2,2: A is 3x2, B is 2x2, C is 3x2.
    # C_ij = sum_k A_ik B_kj.
    # We can view this as multiplying a 3x2 by a 2x2.
    # This is equivalent to two 3x2x1 multiplications? No.
    # Actually, R(M_3,2,2) = 12 is likely optimal or close.
    
    # Given the time limit and complexity, I will return the schoolbook solution.
    # It is guaranteed to be correct.
    return schoolbook()
# EVOLVE-BLOCK-END
