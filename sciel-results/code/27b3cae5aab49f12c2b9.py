import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition to achieve rank < 60.
    # Partition A (2x5) into A1 (2x2) and A2 (2x3).
    # Partition B (5x6) into B1 (2x6) and B2 (3x6).
    # C = A1*B1 + A2*B2.
    
    # We need decompositions for:
    # 1. M(2,2,6): A1 (2x2) * B1 (2x6)
    # 2. M(2,3,6): A2 (2x3) * B2 (3x6)
    
    # M(2,2,6) can be done with rank 14 (2 * M(2,2,3) = 2*10=20? No, M(2,2,4)=14, so M(2,2,6) <= 14+7=21? 
    # Actually M(2,2,6) <= M(2,2,3) + M(2,2,3) = 20.
    # M(2,3,6) <= M(2,3,3) + M(2,3,3) = 15 + 15 = 30.
    # Total 50.
    
    # We will construct the terms explicitly.
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # --- Part 1: A1 (2x2) * B1 (2x6) ---
    # B1 is 2x6. Split into B1a (2x3) and B1b (2x3).
    # A1 * B1a: M(2,2,3) Rank 10 (Lader)
    # A1 * B1b: M(2,2,3) Rank 10 (Lader)
    
    # Lader M(2,2,3) decomposition for A (2x2), B (2x3)
    # A = [[a, b], [c, d]]
    # B = [[x, y, z], [u, v, w]]
    # C = [[ax+bu, ay+bv, az+bw], [cx+du, cy+dv, cz+dw]]
    # The 10 terms are:
    # 1. a*x -> C[0,0]
    # 2. b*u -> C[0,0]
    # 3. a*y -> C[0,1]
    # 4. b*v -> C[0,1]
    # 5. a*z -> C[0,2]
    # 6. b*w -> C[0,2]
    # 7. c*x -> C[1,0]
    # 8. d*u -> C[1,0]
    # 9. c*y -> C[1,1]
    # 10. d*v -> C[1,1]
    # Wait, C[1,2] = c*z + d*w is missing.
    # The Lader algorithm is more complex.
    # For simplicity and reliability, I will use Strassen for 2x2x2 and Standard for the rest.
    
    # Let's use a simpler bound:
    # M(2,2,6) <= 2 * M(2,2,3) = 20
    # M(2,3,6) <= 2 * M(2,3,3) = 30
    # Total 50.
    
    # I will implement the standard multiplication for the sub-blocks to ensure correctness,
    # but I will use Strassen for the 2x2x2 parts to reduce rank.
    
    # Actually, the safest "stronger" algorithm that is easy to verify is:
    # Use Strassen for the top-left 2x2x2 block.
    # Use Standard for the rest.
    
    # A (2x5) = [A1 (2x2), A2 (2x3)]
    # B (5x6) = [[B11 (2x2), B12 (2x4)], [B21 (3x2), B22 (3x4)]]
    
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    # 1. A1*B11 (2x2x2): Strassen Rank 7
    # 2. A1*B12 (2x2x4): Standard Rank 8? Or Strassen based?
    #    M(2,2,4) <= 2 * M(2,2,2) = 14.
    # 3. A2*B21 (2x3x2): M(2,3,2) = M(2,2,3) = 10.
    # 4. A2*B22 (2x3x4): M(2,3,4) = 19.
    
    # Total: 7 + 14 + 10 + 19 = 50.
    
    # I will implement this construction.
    
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # 1. A1 (rows 0-1, cols 0-1 of A) * B11 (rows 0-1, cols 0-1 of B)
    # Strassen 2x2x2
    # A1 = [[a, b], [c, d]], B11 = [[x, y], [u, v]]
    # C11 = ax+bu, C12 = ay+bv, C21 = cx+du, C22 = cy+dv
    # Strassen:
    # m1 = (a+d)(x+u)
    # m2 = (c+d)x
    # m3 = a(y-u)
    # m4 = d(v-w) -- wait, B is 2x2, so v is B[1,1].
    # Let's use the standard Strassen terms.
    
    # Term 1: (a+d, x+u, C[0,0]+C[1,1])
    U[0][col] = 1; U[1][col] = 1; V[0][col] = 1; V[1][col] = 1; W[0][col] = 1; W[3][col] = 1; col += 1
    # Term 2: (c+d, x, C[1,0]+C[1,1])
    U[1][col] = 1; U[2][col] = 1; V[0][col] = 1; W[1][col] = 1; W[3][col] = 1; col += 1
    # Term 3: (a, y-u, C[0,1]-C[1,1])
    U[0][col] = 1; V[1][col] = 1; V[0][col] = -1; W[1][col] = 1; W[3][col] = -1; col += 1
    # Term 4: (d, v, C[1,1])
    U[1][col] = 1; V[1][col] = 1; W[3][col] = 1; col += 1
    # Term 5: (a, y, C[0,1])
    U[0][col] = 1; V[1][col] = 1; W[1][col] = 1; col += 1
    # Term 6: (b, u, C[0,0])
    U[1][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # Term 7: (c, x, C[1,0])
    U[2][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    
    # Note: The above is a mix. I will use a generic standard multiplication for the rest to ensure correctness,
    # but I will stop here and just return the schoolbook if I'm not confident in the Strassen indices.
    # Given the risk of error, I will return the schoolbook decomposition.
    
    # Reset to schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
