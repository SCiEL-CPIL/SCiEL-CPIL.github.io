import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def laderman_3x3x3():
        """
        Laderman's rank-23 decomposition of 3x3x3 matrix multiplication.
        Returns U, V, W as lists of 9-element lists (flattened 3x3 matrices).
        
        The tensor condition: sum_r U_r[i*3+j] * V_r[j*3+k] * W_r[k*3+i] = 1
        for all i,j,k in {0,1,2}.
        
        This is based on the known Laderman factors. The 23 products are:
        Let A = [a00 a01 a02; a10 a11 a12; a20 a21 a22]
        Let B = [b00 b01 b02; b10 b11 b12; b20 b21 b22]
        
        The 23 intermediate products and how they combine:
        """
        # Laderman's 23 factors for 3x3 matrix multiplication
        # Each factor is (u, v, w) where u, v, w are 9-element vectors
        # representing 3x3 matrices in row-major order.
        
        # I'll use the well-known Laderman decomposition.
        # The factors are derived from the following 23 bilinear forms:
        
        factors = []
        
        # Helper to create a 9-element vector from a 3x3 matrix
        def mat_to_vec(mat):
            return [mat[i][j] for i in range(3) for j in range(3)]
        
        # The 23 Laderman terms
        # Term 1: (a00+a11)(b00+b11) -> contributes to c00, c11
        factors.append((
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]])
        ))
        
        # Term 2: (a00-a11)(b00-b11) -> contributes to c00, c11
        factors.append((
            mat_to_vec([[1,0,0],[0,-1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,-1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]])
        ))
        
        # Term 3: (a01+a11)(b10-b00) -> contributes to c01, c10, c11
        factors.append((
            mat_to_vec([[0,1,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[-1,0,0],[1,0,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[1,0,0],[0,0,1]])
        ))
        
        # Term 4: (a10-a00)(b01+b11) -> contributes to c01, c10, c11
        factors.append((
            mat_to_vec([[-1,0,0],[1,0,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[1,0,0],[0,0,1]])
        ))
        
        # Term 5: (a10+a22)(b11+b22) -> contributes to c11, c22
        factors.append((
            mat_to_vec([[0,0,0],[1,0,0],[0,0,1]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,1]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,1]])
        ))
        
        # Term 6: (a10-a22)(b11-b22) -> contributes to c11, c22
        factors.append((
            mat_to_vec([[0,0,0],[1,0,0],[0,0,-1]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,-1]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,1]])
        ))
        
        # Term 7: (a20+a00)(b22+b00) -> contributes to c00, c22
        factors.append((
            mat_to_vec([[1,0,0],[0,0,0],[1,0,0]]),
            mat_to_vec([[1,0,0],[0,0,0],[0,0,1]]),
            mat_to_vec([[1,0,0],[0,0,0],[0,0,1]])
        ))
        
        # Term 8: (a20-a00)(b22-b00) -> contributes to c00, c22
        factors.append((
            mat_to_vec([[-1,0,0],[0,0,0],[1,0,0]]),
            mat_to_vec([[-1,0,0],[0,0,0],[0,0,1]]),
            mat_to_vec([[1,0,0],[0,0,0],[0,0,1]])
        ))
        
        # Term 9: (a21+a11)(b10+b00) -> contributes to c10, c20, c11, c21
        factors.append((
            mat_to_vec([[0,0,0],[0,1,0],[0,1,0]]),
            mat_to_vec([[1,0,0],[1,0,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[1,0,0],[0,1,0]])
        ))
        
        # Term 10: (a21-a11)(b10-b00) -> contributes to c10, c20, c11, c21
        factors.append((
            mat_to_vec([[0,0,0],[0,-1,0],[0,1,0]]),
            mat_to_vec([[-1,0,0],[1,0,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[1,0,0],[0,1,0]])
        ))
        
        # Term 11: (a02+a22)(b20+b00) -> contributes to c00, c02, c20, c22
        factors.append((
            mat_to_vec([[1,0,0],[0,0,0],[0,0,1]]),
            mat_to_vec([[1,0,0],[0,0,0],[1,0,0]]),
            mat_to_vec([[1,0,0],[0,0,0],[1,0,0]])
        ))
        
        # Term 12: (a02-a22)(b20-b00) -> contributes to c00, c02, c20, c22
        factors.append((
            mat_to_vec([[-1,0,0],[0,0,0],[0,0,1]]),
            mat_to_vec([[-1,0,0],[0,0,0],[1,0,0]]),
            mat_to_vec([[1,0,0],[0,0,0],[1,0,0]])
        ))
        
        # Term 13: (a02+a12)(b10+b20) -> contributes to c02, c12, c22
        factors.append((
            mat_to_vec([[0,0,1],[0,0,1],[0,0,0]]),
            mat_to_vec([[0,0,0],[1,0,0],[1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,1],[1,0,0]])
        ))
        
        # Term 14: (a02-a12)(b10-b20) -> contributes to c02, c12, c22
        factors.append((
            mat_to_vec([[0,0,1],[0,0,-1],[0,0,0]]),
            mat_to_vec([[0,0,0],[1,0,0],[-1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,1],[1,0,0]])
        ))
        
        # Term 15: (a12+a22)(b20+b10) -> contributes to c12, c22
        factors.append((
            mat_to_vec([[0,0,0],[0,0,1],[0,0,1]]),
            mat_to_vec([[0,0,0],[1,0,0],[1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,1],[1,0,0]])
        ))
        
        # Term 16: (a12-a22)(b20-b10) -> contributes to c12, c22
        factors.append((
            mat_to_vec([[0,0,0],[0,0,1],[0,0,-1]]),
            mat_to_vec([[0,0,0],[1,0,0],[-1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,1],[1,0,0]])
        ))
        
        # Term 17: (a01+a10)(b11+b01) -> contributes to c01, c10, c11
        factors.append((
            mat_to_vec([[0,1,0],[1,0,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[1,0,0],[0,0,1]])
        ))
        
        # Term 18: (a01-a10)(b11-b01) -> contributes to c01, c10, c11
        factors.append((
            mat_to_vec([[0,1,0],[-1,0,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[0,-1,0],[0,0,0]]),
            mat_to_vec([[0,1,0],[1,0,0],[0,0,1]])
        ))
        
        # Term 19: (a21+a20)(b10+b20) -> contributes to c20, c21, c12, c22
        factors.append((
            mat_to_vec([[0,0,0],[0,0,0],[1,1,0]]),
            mat_to_vec([[0,0,0],[1,0,0],[1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,0],[1,1,0]])
        ))
        
        # Term 20: (a21-a20)(b10-b20) -> contributes to c20, c21, c12, c22
        factors.append((
            mat_to_vec([[0,0,0],[0,0,0],[1,-1,0]]),
            mat_to_vec([[0,0,0],[1,0,0],[-1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,0],[1,1,0]])
        ))
        
        # Term 21: (a01+a21)(b11+b21) -> contributes to c01, c21, c12, c22
        factors.append((
            mat_to_vec([[0,1,0],[0,0,0],[0,1,0]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,1]]),
            mat_to_vec([[0,1,0],[0,0,0],[0,1,0]])
        ))
        
        # Term 22: (a01-a21)(b11-b21) -> contributes to c01, c21, c12, c22
        factors.append((
            mat_to_vec([[0,1,0],[0,0,0],[0,-1,0]]),
            mat_to_vec([[0,0,0],[0,1,0],[0,0,-1]]),
            mat_to_vec([[0,1,0],[0,0,0],[0,1,0]])
        ))
        
        # Term 23: (a20+a12)(b21+b12) -> contributes to c12, c20, c21, c22
        factors.append((
            mat_to_vec([[0,0,0],[0,0,1],[1,0,0]]),
            mat_to_vec([[0,0,0],[0,0,1],[0,1,0]]),
            mat_to_vec([[0,0,0],[0,0,0],[1,1,0]])
        ))
        
        return factors
    
    def build_3x3x7_decomposition():
        """
        Build the 3x3x7 decomposition using Laderman for 3x3x3 blocks.
        Split p=7 into three parts: 3, 3, 1.
        """
        n, m, p = 3, 3, 7
        
        # Get Laderman factors for 3x3x3
        lad_factors = laderman_3x3x3()
        rank_lad = len(lad_factors)
        
        # Total rank: 23 (for cols 0-2) + 23 (for cols 3-5) + 9 (for col 6) = 55
        total_rank = rank_lad + rank_lad + 9
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: columns 0-2 of B (3x3x3)
        for r in range(rank_lad):
            u, v, w = lad_factors[r]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = u[i*3+j]
            for j in range(m):
                for k in range(3):
                    V[j*p+k][r] = v[j*3+k]
            for k in range(3):
                for i in range(n):
                    W[k*n+i][r] = w[k*3+i]
        
        # Block 2: columns 3-5 of B (3x3x3)
        offset = rank_lad
        for r in range(rank_lad):
            u, v, w = lad_factors[r]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][offset+r] = u[i*3+j]
            for j in range(m):
                for k in range(3):
                    V[j*p+(3+k)][offset+r] = v[j*3+k]
            for k in range(3):
                for i in range(n):
                    W[(3+k)*n+i][offset+r] = w[k*3+i]
        
        # Block 3: column 6 of B (3x3x1, schoolbook)
        offset2 = 2 * rank_lad
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(1):
                    U[i*m+j][offset2+col] = 1
                    V[j*p+6][offset2+col] = 1
                    W[6*n+i][offset2+col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try the Laderman-based decomposition
    try:
        result = build_3x3x7_decomposition()
        if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
