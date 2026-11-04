"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Construct a rank 28 decomposition using Strassen for 2x2 blocks
    # and schoolbook for the rest.
    # Structure:
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x5) = [B1 (2x5); B2 (1x5)]
    # C = A1 B1 + A2 B2
    
    # B1 (2x5) is split into:
    # B1a (2x2) cols 0,1
    # B1b (2x2) cols 2,3
    # B1c (2x1) col 4
    
    # Ranks:
    # A2 B2 (2x1 x 1x5) -> rank 10 (schoolbook)
    # A1 B1a (2x2 x 2x2) -> rank 7 (Strassen)
    # A1 B1b (2x2 x 2x2) -> rank 7 (Strassen)
    # A1 B1c (2x2 x 2x1) -> rank 4 (schoolbook)
    # Total rank = 10 + 7 + 7 + 4 = 28
    
    R = 28
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for i in u_indices:
            U[i][col] += 1
        for j in v_indices:
            V[j][col] += 1
        for k in w_indices:
            W[k][col] += 1
        col += 1
        
    # Helper to add a Strassen term
    # u_map: dict from (i,j) to coeff
    # v_map: dict from (j,k) to coeff
    # w_map: dict from (k,i) to coeff
    def add_strassen_term(u_map, v_map, w_map):
        nonlocal col
        for (i, j), c in u_map.items():
            if c != 0:
                U[i*m+j][col] += c
        for (j, k), c in v_map.items():
            if c != 0:
                V[j*p+k][col] += c
        for (k, i), c in w_map.items():
            if c != 0:
                W[k*n+i][col] += c
        col += 1

    # 1. A2 B2 (j=2)
    # A2 indices: i=0,1; j=2. U rows: 2, 5
    # B2 indices: j=2; k=0..4. V rows: 10..14
    # C indices: k=0..4; i=0,1. W rows: k*2+i
    for i in range(2):
        for k in range(5):
            add_term([i*m+2], [2*p+k], [k*n+i])
            
    # 2. A1 B1a (j=0,1; k=0,1)
    # A1 indices: i=0,1; j=0,1. U rows: 0,1,2,3
    # B1a indices: j=0,1; k=0,1. V rows: 0,1,5,6
    # C indices: k=0,1; i=0,1. W rows: 0,1,2,3
    
    # Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    add_strassen_term(
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1} # C00, C11, -C01, -C10? 
        # Wait, W indices are (k,i). 
        # C00 is k=0,i=0 -> W[0]
        # C11 is k=1,i=1 -> W[3]
        # C01 is k=1,i=0 -> W[2]
        # C10 is k=0,i=1 -> W[1]
        # Formula: C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So for M1: W[0]=1, W[3]=1. W[1], W[2] are 0.
    )
    # Correction: I need to be careful with the W indices for Strassen.
    # Let's re-define the Strassen terms properly.
    
    # Reset col for Strassen part? No, just continue.
    # But I already added M1 incorrectly above. Let's restart the Strassen part logic.
    # I will clear the last column and redo it properly.
    col -= 1
    for r in range(n*m): U[r][col] = 0
    for r in range(m*p): V[r][col] = 0
    for r in range(p*n): W[r][col] = 0
    
    # Strassen Terms for B1a (k=0,1)
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    add_strassen_term(
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}
    )
    # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    add_strassen_term(
        {(1,0):1, (1,1):1}, 
        {(0,0):1}, 
        {(0,1):1, (1,1):-1}
    )
    # M3: A00(B10-B00) -> C01 += 1, C11 += 1
    add_strassen_term(
        {(0,0):1}, 
        {(1,0):1, (0,0):-1}, 
        {(1,0):1, (1,1):1}
    )
    # M4: A11(B11-B01) -> C00 += 1, C10 += 1
    add_strassen_term(
        {(1,1):1}, 
        {(1,1):1, (0,1):-1}, 
        {(0,0):1, (0,1):1}
    )
    # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    add_strassen_term(
        {(0,0):1, (0,1):1}, 
        {(1,1):1}, 
        {(0,0):-1, (1,0):1}
    )
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    add_strassen_term(
        {(1,0):1, (0,0):-1}, 
        {(0,0):1, (0,1):1}, 
        {(1,1):1}
    )
    # M7: (A01-A11)(B10+B11) -> C00 += 1
    add_strassen_term(
        {(0,1):1, (1,1):-1}, 
        {(1,0):1, (1,1):1}, 
        {(0,0):1}
    )
    
    # 3. A1 B1b (j=0,1; k=2,3)
    # B1b indices: j=0,1; k=2,3. V rows: 2,3,7,8
    # C indices: k=2,3; i=0,1. W rows: 4,5,6,7
    
    # M1b: (A00+A11)(B02+B13) -> C20 += 1, C31 += 1
    add_strassen_term(
        {(0,0):1, (1,1):1}, 
        {(0,2):1, (1,3):1}, 
        {(2,0):1, (3,1):1}
    )
    # M2b: (A10+A11)B02 -> C21 += 1, C31 -= 1
    add_strassen_term(
        {(1,0):1, (1,1):1}, 
        {(0,2):1}, 
        {(2,1):1, (3,1):-1}
    )
    # M3b: A00(B12-B02) -> C30 += 1, C31 += 1
    add_strassen_term(
        {(0,0):1}, 
        {(1,2):1, (0,2):-1}, 
        {(3,0):1, (3,1):1}
    )
    # M4b: A11(B13-B03) -> C20 += 1, C21 += 1
    add_strassen_term(
        {(1,1):1}, 
        {(1,3):1, (0,3):-1}, 
        {(2,0):1, (2,1):1}
    )
    # M5b: (A00+A01)B13 -> C20 -= 1, C30 += 1
    add_strassen_term(
        {(0,0):1, (0,1):1}, 
        {(1,3):1}, 
        {(2,0):-1, (3,0):1}
    )
    # M6b: (A10-A00)(B02+B03) -> C31 += 1
    add_strassen_term(
        {(1,0):1, (0,0):-1}, 
        {(0,2):1, (0,3):1}, 
        {(3,1):1}
    )
    # M7b: (A01-A11)(B12+B13) -> C20 += 1
    add_strassen_term(
        {(0,1):1, (1,1):-1}, 
        {(1,2):1, (1,3):1}, 
        {(2,0):1}
    )
    
    # 4. A1 B1c (j=0,1; k=4)
    # B1c indices: j=0,1; k=4. V rows: 4,9
    # C indices: k=4; i=0,1. W rows: 8,9
    
    # Schoolbook for 2x2x1
    # C40 = A00 B04 + A01 B14
    # C41 = A10 B04 + A11 B14
    
    add_term([0], [4], [8]) # A00 B04 -> C40
    add_term([1], [9], [8]) # A01 B14 -> C40
    add_term([2], [4], [9]) # A10 B04 -> C41
    add_term([3], [9], [9]) # A11 B14 -> C41
    
    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                expected = 1 if (j == jp and ip == i and kp == k) else 0
                                actual = 0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[jp*p+kp][r] * W[k*n+ip][r]
                                if actual != expected:
                                    return False
        return True
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        R = rank_schoolbook

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
