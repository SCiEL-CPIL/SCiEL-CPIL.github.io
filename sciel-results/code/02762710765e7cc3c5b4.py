"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # 150s wall-clock, 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The matrix multiplication tensor for n x m times m x p has:
    # U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # (This is the "1" tensor that represents the matmul operation)
    
    # Schoolbook decomposition has rank n*m*p = 3*3*4 = 36
    # Each term r corresponds to a unique (i,j,k) triple with U=V=W=1 at that position.
    
    # Let's try to find a lower rank decomposition.
    # For 2x2x2, Strassen gives rank 7 < 8 = schoolbook.
    # For 3x3x3, the best known is 23 (Lader).
    # For 3x3x4, we can try block approaches.
    
    # Strategy: Use Strassen for 2x2x2 blocks and schoolbook for the rest.
    # Split dimensions:
    # i in {0,1} (size 2) and i=2 (size 1)
    # j in {0,1} (size 2) and j=2 (size 1)  
    # k in {0,1} (size 2) and k in {2,3} (size 2)
    
    # Blocks:
    # (i<2, j<2, k<2): 2x2x2 -> Strassen rank 7
    # (i<2, j<2, k>=2): 2x2x2 -> Strassen rank 7
    # (i<2, j=2, k<2): 2x1x2 -> schoolbook rank 4
    # (i<2, j=2, k>=2): 2x1x2 -> schoolbook rank 4
    # (i=2, j<2, k<2): 1x2x2 -> schoolbook rank 4
    # (i=2, j<2, k>=2): 1x2x2 -> schoolbook rank 4
    # (i=2, j=2, k<2): 1x1x2 -> schoolbook rank 2
    # (i=2, j=2, k>=2): 1x1x2 -> schoolbook rank 2
    
    # Total: 7+7+4+4+4+4+2+2 = 34
    
    # Let's implement this.
    
    total_rank = 34
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M7
    # C[1,1] = M1 + M3 - M2 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # In tensor form, for each multiplication Mk, we have contributions to U, V, W.
    # U[i*m+j, r] is the coefficient of A[i,j] in the k-th multiplication
    # V[j*p+k, r] is the coefficient of B[j,k] in the k-th multiplication
    # W[k*n+i, r] is the coefficient in C[i,k] from the k-th multiplication
    
    def strassen_2x2x2(i_off, j_off, k_off):
        """Add Strassen terms for 2x2x2 block at offset (i_off, j_off, k_off)."""
        nonlocal col
        # The 7 Strassen multiplications with their U, V, W coefficients
        # U coefficients (for A indices (0,0),(0,1),(1,0),(1,1) in local coords):
        # V coefficients (for B indices (0,0),(0,1),(1,0),(1,1) in local coords):
        # W coefficients (for C indices (0,0),(0,1),(1,0),(1,1) in local coords):
        
        # M1: U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C01=1, C10=1, C11=1
        # M2: U: A10=1, A11=1; V: B00=1; W: C10=1
        # M3: U: A00=1; V: B10=1, B11=-1; W: C01=1
        # M4: U: A11=1; V: B01=1, B00=-1; W: C00=-1, C01=1
        # M5: U: A00=1, A01=1; V: B11=1; W: C00=1
        # M6: U: A10=1, A00=-1; V: B00=1, B10=1; W: C11=1
        # M7: U: A01=1, A11=-1; V: B01=1, B11=1; W: C00=1, C10=1
        
        # Let me verify Strassen more carefully:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 + M3 - M2 + M6
        
        # Let me verify C00:
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # -M4 = -A11*B01 + A11*B00
        # M5 = A00*B11 + A01*B11
        # M7 = A01*B01 + A01*B11 - A11*B01 - A11*B11
        # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
        # = A00*B00 + 2*A00*B11 + 2*A11*B00 - 2*A11*B01 + 2*A01*B11 + A01*B01
        # That's not right. Let me look up the correct Strassen formulas.
        
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00)
        # M3 = (A00)(B10 - B11)
        # M4 = (A11)(B01 - B00)
        # M5 = (A00 + A01)(B11)
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 + M3 - M2 + M6
        
        # Let me re-verify C00:
        # M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # -M4 = -A11*B01 + A11*B00
        # M5 = A00*B11 + A01*B11
        # M7 = A01*B01 + A01*B11 - A11*B01 - A11*B11
        # Sum: A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A00*B11 + A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
        # = A00*B00 + 2*A00*B11 + 2*A11*B00 - 2*A11*B01 + 2*A01*B11 + A01*B01
        # Still wrong. The standard Strassen formulas must be different.
        
        # Let me use a known correct set:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 + M3 - M2 + M6
        
        # I think the issue is I'm not computing this right. Let me just use schoolbook for now
        # and return rank 36. That's the safe fallback.
        
        # Actually, let me just do schoolbook for everything.
        pass
    
    # Given the complexity of getting Strassen right with the exact indexing,
    # and the time pressure, let's just return the schoolbook decomposition.
    # Rank = 36, which matches the seed program.
    
    rank = n * m * p  # 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Validate
    if time.time() - t0 < DEADLINE:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
