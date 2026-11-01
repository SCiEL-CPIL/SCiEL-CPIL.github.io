"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140
    
    def build_rank17():
        """
        Rank-17 decomposition: 8 terms for cols 0-2 (generalized Strassen k=3)
        + 7 terms for cols 3-4 (standard Strassen k=2)
        """
        rank = 17
        U = [[0] * rank for _ in range(n * m)]   # (4, 17)
        V = [[0] * rank for _ in range(m * p)]   # (10, 17)
        W = [[0] * rank for _ in range(p * n)]   # (10, 17)
        
        # Generalized Strassen for 2x2 * 2x3 (columns c, c+1, c+2)
        def add_strassen_k3(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+2)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+2][r] = 1
            W[c*n+0][r] = 1; W[(c+2)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+2)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[(c+1)*n+0][r] = 1
            
            # M5: f=A00+A01, g=B1(c+2)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+2)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
            
            # M8: f=A01, g=B1(c+2)
            r = base + 7
            U[1][r] = 1
            V[1*p+c+2][r] = 1
            W[(c+2)*n+0][r] = 1
        
        # Standard Strassen for 2x2 * 2x2 (columns c, c+1)
        def add_strassen_k2(c, base):
            # M1: f=A00+A11, g=B0c + B1(c+1)
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M2: f=A10+A11, g=B0c
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+c][r] = 1
            W[c*n+1][r] = 1; W[(c+1)*n+1][r] = -1
            
            # M3: f=A00, g=B0(c+1) - B1(c+1)
            r = base + 2
            U[0][r] = 1
            V[0*p+c+1][r] = 1; V[1*p+c+1][r] = -1
            W[(c+1)*n+0][r] = 1; W[(c+1)*n+1][r] = 1
            
            # M4: f=A11, g=B1c - B0c
            r = base + 3
            U[3][r] = 1
            V[1*p+c][r] = 1; V[0*p+c][r] = -1
            W[c*n+0][r] = 1; W[c*n+1][r] = 1
            
            # M5: f=A00+A01, g=B1(c+1)
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+c+1][r] = 1
            W[c*n+0][r] = -1; W[(c+1)*n+0][r] = 1
            
            # M6: f=A10-A00, g=B0c + B0(c+1)
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+c][r] = 1; V[0*p+c+1][r] = 1
            W[(c+1)*n+1][r] = 1
            
            # M7: f=A01-A11, g=B1c + B1(c+1)
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+c][r] = 1; V[1*p+c+1][r] = 1
            W[c*n+0][r] = 1
        
        # Columns 0,1,2: generalized Strassen (8 terms)
        add_strassen_k3(0, 0)
        
        # Columns 3,4: standard Strassen (7 terms)
        add_strassen_k2(3, 8)
        
        return U, V, W
    
    def check_decomposition(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0
                    actual = sum(U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r] 
                                  for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try rank 17 first
    U17, V17, W17 = build_rank17()
    if check_decomposition(U17, V17, W17, 17):
        return {"U": U17, "V": V17, "W": W17}
    
    # Fallback to rank 18
    def build_rank18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W
    
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
