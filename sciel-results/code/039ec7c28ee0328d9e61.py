import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 130.0

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def make_strassen_3x3x5():
        """
        Construct a rank-44 decomposition for 3x3x5 matrix multiplication.
        
        Strategy: Partition the 3x3 matrix into blocks and use Strassen for the 2x2 sub-block,
        then schoolbook for the remaining parts.
        
        For 3x3x3, we can use a rank-7 Strassen-like decomposition.
        For 3x3x5, we decompose into 5 independent 3x3 multiplications.
        
        A better approach: Use the known rank-44 algorithm for 3x3x5.
        """
        # Rank 44 construction using structured decomposition
        # We'll use a hybrid approach: Strassen for 2x2 blocks + schoolbook for edges
        
        rank = 44
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(15)]
        W = [[0] * rank for _ in range(15)]
        
        # For each of the 5 columns k in the 3x5 matrix:
        # We multiply 3x3 @ 3x1 (column k)
        # Use Strassen-like decomposition for each 3x3 multiplication
        
        # Actually, let's use a cleaner approach:
        # The tensor for 3x3x5 has 45 entries (3*3*5)
        # We need rank 44, so we save 1 rank
        
        # Use the following construction:
        # For k=0,1,2,3: use rank-9 schoolbook (total 36)
        # For k=4: use rank-8 (save 1 rank)
        # Total: 36 + 8 = 44
        
        # Schoolbook for k=0,1,2,3 (4 columns * 9 products = 36 ranks)
        idx = 0
        for k in range(4):
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j][idx] = 1
                    V[j * 5 + k][idx] = 1
                    W[k * 3 + i][idx] = 1
                    idx += 1
        
        # For k=4, use rank-8 decomposition
        # This is a 3x3x1 multiplication (matrix-vector product)
        # Standard rank is 9, but we can save 1 rank
        
        # For 3x3x1, the tensor is sum_{i,j} A[i,j] * B[j,4] * C[4,i]
        # We can use: C[4,i] = sum_j A[i,j] * B[j,4]
        # This is just a matrix-vector product, which has rank 9 in general
        
        # Actually, for a single column, we need all 9 products.
        # Let me reconsider the decomposition.
        
        # Alternative: Use Strassen for a 2x2 sub-block
        # Partition 3x3 into:
        # [A11 A12 A13]
        # [A21 A22 A23]
        # [A31 A32 A33]
        
        # Use Strassen on the top-left 2x2 block (ranks 1-7)
        # Schoolbook for the rest (ranks 8-44)
        
        # Reset
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(15)]
        W = [[0] * rank for _ in range(15)]
        
        # Strassen for 2x2 sub-block (rows 0-1, cols 0-1)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # For each column k in 0..4:
        for k in range(5):
            # M1 = (A00+A11)(B0k+B1k)
            U[0 * 3 + 0][idx] = 1  # A00
            U[1 * 3 + 1][idx] = 1  # A11
            V[0 * 5 + k][idx] = 1  # B0k
            V[1 * 5 + k][idx] = 1  # B1k
            W[k * 3 + 0][idx] = 1  # C00
            W[k * 3 + 1][idx] = 1  # C01
            W[k * 3 + 1][idx] = -1  # C10
            W[k * 3 + 0][idx] = -1  # C11
            # This is getting complicated. Let me use a simpler approach.
            idx += 1
        
        # This approach is getting too complex. Let me use a known rank-44 construction.
        
        # Actually, the simplest rank-44 construction:
        # Use schoolbook for all 45 products, but merge one pair.
        # For example, for k=0: merge A00*B00 and A01*B01 into one term
        # This doesn't work directly.
        
        # Let me just return schoolbook rank 45 for now, and try to optimize.
        return None, None, None, 45

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    def try_strassen_hybrid():
        """
        Try a hybrid Strassen + schoolbook approach for rank 44.
        
        Partition the 3x3 matrix as:
        [A11 A12 | A13]
        [A21 A22 | A23]
        [--------+----]
        [A31 A32 | A33]
        
        Use Strassen for the 2x2 block (A11:A22) with each column of B.
        Use schoolbook for the third row and third column.
        
        For each column k of B (5 columns):
        - Strassen on 2x2 block: 7 ranks
        - Schoolbook for A31*B1k, A32*B2k, A33*B3k: 3 ranks
        - Schoolbook for A13*B3k, A23*B3k: 2 ranks
        Total per column: 7 + 3 + 2 = 12 ranks
        Total for 5 columns: 60 ranks - too many.
        
        Let me try a different partition.
        """
        rank = 44
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(15)]
        W = [[0] * rank for _ in range(15)]
        
        # Use Strassen for 2x2 sub-blocks in a clever way
        # Actually, let me use the known result that 3x3x3 has rank 7
        # and construct 3x3x5 by combining 5 such decompositions
        
        # For each column k (0..4), we need to compute C[:,k] = A @ B[:,k]
        # This is a 3x3x1 multiplication
        
        # The 3x3x1 tensor has rank 9 in general (matrix-vector product)
        # But we can use a rank-8 decomposition if we share structure
        
        # Let me just use schoolbook for now
        return None, None, None, 45

    best_U, best_V, best_W, best_rank = None, None, None, 45
    
    # Try to find rank-44 decomposition
    # For now, fall back to schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
