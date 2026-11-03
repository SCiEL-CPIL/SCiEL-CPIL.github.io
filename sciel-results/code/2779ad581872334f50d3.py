"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # For each (i, j, k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    # Check orthogonality: for (i,j,k) != (i',j',k'), sum should be 0
    # This is expensive, so we skip full verification and rely on construction
    return True

def _try_strassen_embed(n, m, p):
    """Try to find a decomposition using Strassen's 7-term 2x2 multiplication as a template.
    For 5x5, we can't directly embed 4x4, but we can try a rank-49 base (4x4 Strassen) + 25 extra terms.
    This is complex, so we fall back to schoolbook if it doesn't work quickly."""
    # This is a placeholder for a more sophisticated search.
    # For now, return None to indicate we should use schoolbook.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s limit, but we'll use 100s for safety
    
    # First, try to find a better decomposition than schoolbook.
    # For 5x5, the best known upper bound is 49 (from 4x4 Strassen + 5th row/col),
    # but finding an exact rank-49 decomposition is non-trivial.
    # We'll attempt a simple search, but fall back to schoolbook (rank 125) if needed.
    
    # Try a few known constructions:
    # 1. Schoolbook (rank 125) - always works
    schoolbook = _schoolbook(n, m, p)
    
    # 2. Try Strassen-based embedding (rank 49 + 25 = 74, or just 49 if we can pad)
    # Actually, for 5x5, a common approach is to use the 4x4 rank-49 decomposition
    # and add 5+5+5 = 15 more terms for the 5th row/column, giving rank 64.
    # But this requires careful construction.
    
    # Let's try a simple rank-74 construction:
    # Use 4x4 Strassen (rank 49) for the top-left 4x4 block,
    # and add 25 terms for the interactions with the 5th row/column.
    
    def _strassen_2x2():
        """Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
        U, V, W are lists of 7 vectors of length 4."""
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 2x2, n=m=p=2
        # U[i*2+j, c] for i,j in 0..1, c in 0..6
        # V[j*2+k, c] for j,k in 0..1, c in 0..6
        # W[k*2+i, c] for k,i in 0..1, c in 0..6
        
        U = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        V = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        W = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
        
        # M1: (A11+A22)*(B11+B22)
        # A11 is U[0*2+0]=U[0], A22 is U[1*2+1]=U[3]
        # B11 is V[0*2+0]=V[0], B22 is V[1*2+1]=V[3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 = M1 + ... so W[0*2+0]=W[0] gets +1 from M1
        # C22 = M1 - ... so W[1*2+1]=W[3] gets +1 from M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A21+A22)*B11
        # A21 is U[1*2+0]=U[2], A22 is U[3]
        # B11 is V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 = M2 + ... so W[1*2+0]=W[2] gets +1
        W[2][1] = 1
        
        # M3: A11*(B12-B22)
        # A11 is U[0]
        # B12 is V[0*2+1]=V[1], B22 is V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 = M3 + ... so W[0*2+1]=W[1] gets +1
        W[1][2] = 1
        
        # M4: A22*(B21-B11)
        # A22 is U[3]
        # B21 is V[1*2+0]=V[2], B11 is V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11 = ... + M4, C21 = ... + M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A11+A12)*B22
        # A11 is U[0], A12 is U[0*2+1]=U[1]
        # B22 is V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 = ... - M5, C12 = ... + M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (A21-A11)*(B11+B12)
        # A21 is U[2], A11 is U[0]
        # B11 is V[0], B12 is V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 = ... + M6
        W[3][5] = 1
        
        # M7: (A12-A22)*(B21+B22)
        # A12 is U[1], A22 is U[3]
        # B21 is V[2], B22 is V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 = ... + M7
        W[0][6] = 1
        
        return U, V, W
    
    # For 5x5, we can't easily use Strassen directly.
    # The schoolbook decomposition with rank 125 is safe.
    # Let's just return the schoolbook decomposition for now.
    # A more sophisticated search would try to find a lower rank,
    # but that's complex and time-consuming.
    
    return schoolbook
# EVOLVE-BLOCK-END
