"""Matrix multiplication 2x2x9 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    def schoolbook():
        """Rank 36 schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify tensor identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2x2 (rank 7).
        Returns U (4x7), V (4x7), W (4x7)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # c=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # c=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        # c=2: M3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1

        # c=3: M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1

        # c=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1

        # c=5: M6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[2][5] = 1; W[3][5] = 1

        # c=6: M7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def strassen_2x2x1():
        """
        Schoolbook decomposition for 2x2x1 (rank 4).
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]

        # c=0: a00*b00
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1

        # c=1: a01*b10
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1

        # c=2: a10*b00
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1

        # c=3: a11*b10
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1

        return U, V, W

    def build_strassen_block_decomp():
        """
        Build rank-32 decomposition: 4 Strassen 2x2x2 blocks + 1 schoolbook 2x2x1 block.
        """
        U_s, V_s, W_s = strassen_2x2x2()
        U_1, V_1, W_1 = strassen_2x2x1()

        rank = 7 * 4 + 4  # 32
        U_full = [[0] * rank for _ in range(n * m)]  # 4 x 32
        V_full = [[0] * rank for _ in range(m * p)]  # 18 x 32
        W_full = [[0] * rank for _ in range(p * n)]  # 18 x 32

        col_offset = 0
        for block in range(4):
            for c in range(7):
                for i in range(4):
                    U_full[i][col_offset + c] = U_s[i][c]

                for j in range(2):
                    for k_local in range(2):
                        k_global = 2 * block + k_local
                        V_full[j * p + k_global][col_offset + c] = V_s[j * 2 + k_local][c]

                for k_local in range(2):
                    for i in range(2):
                        k_global = 2 * block + k_local
                        W_full[k_global * n + i][col_offset + c] = W_s[k_local * 2 + i][c]

            col_offset += 7

        # Last block: schoolbook 2x2x1 for column 8
        for c in range(4):
            for i in range(4):
                U_full[i][col_offset + c] = U_1[i][c]

            for j in range(2):
                V_full[j * p + 8][col_offset + c] = V_1[j][c]

            for i in range(2):
                W_full[8 * n + i][col_offset + c] = W_1[i][c]

        return {"U": U_full, "V": V_full, "W": W_full}

    def try_dft_based_decomp():
        """
        Structural change: attempt a DFT-based decomposition exploiting the cyclic structure of p=9.
        
        The idea: decompose the 9-dimensional k-index into cyclic subgroups.
        Since 9 = 3*3, we can try to use the structure Z_9 ≅ Z_3 × Z_3.
        
        For each (i,j) pair, the dependence on k is through the tensor entries.
        If we can find a low-rank decomposition of the k-dependence using DFT-like basis,
        we might reduce the overall rank.
        
        However, for matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k.
        This means we need:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        The DFT approach would try to express the constant function 1 as a product of 
        lower-dimensional functions in a decomposed form.
        
        Let's try a different structural approach: exploit the fact that for fixed i,j,
        the k-dependence in V and W can be shared.
        
        Actually, let's try a more concrete approach:
        Split k into k = 3*a + b where a,b in {0,1,2}.
        Then we have a 2x2x3 (in a) × 2x2x3 (in b) structure.
        
        For the a-dimension (3 values), we can use a rank-3 decomposition (schoolbook for 2x2x3 is 12,
        but maybe we can do better with the b-dimension sharing).
        
        This is complex, so let's try a simpler structural variant:
        Use the fact that the 9 k-values can be grouped into 3 groups of 3,
        and use a shared decomposition across groups.
        
        For now, let's try to find a decomposition with rank < 32 by exploiting
        the symmetry in the k-index.
        
        One approach: use the fact that for each (i,j), the vector (V[j*p+k,c] for k=0..8)
        and (W[k*n+i,c] for k=0..8) have structure.
        
        Let's try a rank-28 decomposition by sharing some factors across k-groups.
        """
        # This is a placeholder for a more sophisticated DFT-based approach.
        # For now, return None to indicate we should fall back to the block decomposition.
        return None

    def try_reduced_rank_search():
        """
        Try to find a decomposition with rank < 32 using a heuristic search.
        
        Start from the rank-32 block decomposition and try to merge columns
        that are linearly dependent or can be combined.
        """
        # This would require a more sophisticated search algorithm.
        # For now, return None.
        return None

    # Try the Strassen block decomposition (rank 32)
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = build_strassen_block_decomp()
            if verify(result["U"], result["V"], result["W"]):
                return result
        except Exception:
            pass

    # Try DFT-based decomposition
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            result = try_dft_based_decomp()
            if result is not None and verify(result["U"], result["V"], result["W"]):
                return result
        except Exception:
            pass

    # Try reduced rank search
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = try_reduced_rank_search()
            if result is not None and verify(result["U"], result["V"], result["W"]):
                return result
        except Exception:
            pass

    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
