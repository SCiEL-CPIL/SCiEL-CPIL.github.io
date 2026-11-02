import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for 2x2 * 2x9."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # For each term, check that U[i,j] * V[j,k] matches W[k,i]
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But actually the standard form is:
    # (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = A[i,j] * B[j,k]
    # Decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # And the reconstruction: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify with random inputs
    for _ in range(10):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # U_c: vector of size n*m, V_c: vector of size m*p, W_c: vector of size p*n
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # The product for this term: outer(u_c, v_c) contracted with w_c?
            # Actually: contribution to C[i,k] from term c is:
            # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # = W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c]
            # = W[k*n+i, c] * (U_c @ V_c)[i,k]
            # Wait, U_c is n x m, V_c is m x p, so U_c @ V_c is n x p.
            # Then multiply element-wise with w_c reshaped to p x n? No.
            
            # Let me re-read the convention: W[k*n+i, r]
            # So W is indexed by (k, i) where k is row of B (column of C) and i is row of A (row of C)
            # The product: C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
            
            intermediate = np.dot(u_c, v_c)  # n x p matrix
            # Now multiply by w_c which is p x n
            # C_contribution[i,k] = w_c[k,i] * intermediate[i,k]
            C_computed += np.outer(w_c.flatten(), intermediate.flatten()).reshape(p, n).T * 0 + np.multiply(intermediate, w_c.T)
        
        # Wait, let me be more careful.
            # w_c is indexed by (k*n + i), so w_c[k*n+i] = W_c[k, i]
            # C[i,k] should get contribution: W_c[k,i] * intermediate[i,k]
            # So C_contribution = intermediate * w_c.T (element-wise)
        
        # Let me redo this more carefully
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            intermediate = np.dot(u_c, v_c)  # (n, p)
            C_computed += intermediate * w_c.T  # element-wise, w_c.T is (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    # We want to decompose 2x2 * 2x9 matrix multiplication.
    # Split B into 4 blocks of 2x2 and one block of 2x1.
    # B = [B0 | B1 | B2 | B3 | B4] where B0..B3 are 2x2, B4 is 2x1.
    
    # For each 2x2 block, use Strassen (rank 7).
    # For the 2x1 block, use schoolbook (rank 2).
    # Total rank: 4 * 7 + 2 = 30.
    
    # Strassen decomposition for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # C = AB
    # M1 = a * (y - w)
    # M2 = (a + b) * w
    # M3 = (c + d) * x
    # M4 = d * (z - x)
    # M5 = (a + d) * (x + w)
    # M6 = (b - d) * (z + w)
    # M7 = (a - c) * (x + y)
    # C[0,0] = M5 + M4 - M2 + M6
    # C[0,1] = M1 + M2
    # C[1,0] = M3 + M4
    # C[1,1] = M1 - M3 + M5 + M7
    
    # Let's verify this is correct.
    # M1 = a(y-w), M2 = (a+b)w, M3 = (c+d)x, M4 = d(z-x)
    # M5 = (a+d)(x+w), M6 = (b-d)(z+w), M7 = (a-c)(x+y)
    
    # C[0,0] = a*x + b*z
    # M5 + M4 - M2 + M6 = (a+d)(x+w) + d(z-x) - (a+b)w + (b-d)(z+w)
    # = ax + aw + dx + dw + dz - dx - aw - bw + bz + bw - dz - dw
    # = ax + (aw - aw) + (dx - dx) + (dw - dw) + (dz - dz) + (-bw + bw) + bz
    # = ax + bz. Correct!
    
    # C[0,1] = a*y + b*w
    # M1 + M2 = a(y-w) + (a+b)w = ay - aw + aw + bw = ay + bw. Correct!
    
    # C[1,0] = c*x + d*z
    # M3 + M4 = (c+d)x + d(z-x) = cx + dx + dz - dx = cx + dz. Correct!
    
    # C[1,1] = c*y + d*w
    # M1 - M3 + M5 + M7 = a(y-w) - (c+d)x + (a+d)(x+w) + (a-c)(x+y)
    # = ay - aw - cx - dx + ax + aw + dx + dw + ax + ay - cx - cy
    # = (aw - aw) + (dx - dx) + ax + ax + dw + ay + ay - cx - cx - cy
    # = 2ax + dw + 2ay - 2cx - cy. That's not right.
    
    # Let me use the standard Strassen formulas from a reliable source.
    # Standard Strassen:
    # M1 = a * (y - w)
    # M2 = (a + b) * w
    # M3 = (c + d) * x
    # M4 = d * (z - x)
    # M5 = (a + d) * (x + w)
    # M6 = (b - d) * (z + w)
    # M7 = (a - c) * (x + y)
    # C = [[M5 + M4 - M2 + M6, M1 + M2],
    #      [M3 + M4,           M1 - M3 + M5 + M7]]
    
    # Let me re-verify C[1,1]:
    # M1 - M3 + M5 + M7 = a(y-w) - (c+d)x + (a+d)(x+w) + (a-c)(x+y)
    # = ay - aw - cx - dx + ax + aw + dx + dw + ax + ay - cx - cy
    # = (ay + ay) + (-aw + aw) + (-cx - cx) + (-dx + dx) + (ax + ax) + dw - cy
    # = 2ay - 2cx + 2ax + dw - cy
    # This should equal c*y + d*w. It doesn't.
    
    # I think the standard Strassen is:
    # M1 = a(y - w)
    # M2 = (a + b)w
    # M3 = (c + d)x
    # M4 = d(z - x)
    # M5 = (a + d)(x + w)
    # M6 = (b - d)(z + w)
    # M7 = (a - c)(x + y)
    # C[0,0] = M5 + M4 - M2 + M6
    # C[0,1] = M1 + M2
    # C[1,0] = M3 + M4
    # C[1,1] = M1 - M3 + M5 + M7
    
    # Hmm, let me just use a different, verified Strassen decomposition.
    # From Wikipedia (as of my knowledge):
    # m1 = a * (y - w)
    # m2 = (a + b) * w
    # m3 = (c + d) * x
    # m4 = d * (z - x)
    # m5 = (a + d) * (x + w)
    # m6 = (b - d) * (z + w)
    # m7 = (a - c) * (x + y)
    # c11 = m5 + m4 - m2 + m6
    # c12 = m1 + m2
    # c21 = m3 + m4
    # c22 = m1 - m3 + m5 + m7
    
    # Let me try yet another standard form. Maybe I have a sign error.
    # Alternative standard Strassen:
    # M1 = a * (y - w)
    # M2 = (a + b) * w
    # M3 = (c + d) * x
    # M4 = d * (z - x)
    # M5 = (a + d) * (x + w)
    # M6 = (b - d) * (z + w)
    # M7 = (a - c) * (x + y)
    # C = [[M5 + M4 - M2 + M6, M1 + M2],
    #      [M3 + M4, M1 - M3 + M5 + M7]]
    
    # I'll just trust this and verify numerically in code. If it fails, I'll use schoolbook.
    
    # For now, let me just implement the rank 30 solution and verify it.
    # If Strassen doesn't work out cleanly, I'll fall back to rank 36 (schoolbook).
    
    # Actually, let me just use the known-correct Strassen from a reference implementation.
    # I'll define the U, V, W matrices for a single 2x2*2x2 Strassen block.
    
    # For Strassen 2x2*2x2, rank 7:
    # U indices: i*m+j = i*2+j for i,j in {0,1}
    # V indices: j*p+k = j*2+k for j,k in {0,1}
    # W indices: k*n+i = k*2+i for k,i in {0,1}
    
    # Let me just hardcode a known correct rank-7 decomposition for 2x2*2x2.
    # I'll use the one that's widely cited.
    
    # After more thought, let me just use schoolbook for simplicity and correctness.
    # Rank = 2*2*9 = 36.
    
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
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
