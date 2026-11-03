import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split m=3 into m1=2 and m2=1.
    # T_{2,3,8} = T_{2,2,8} + T_{2,1,8}
    # T_{2,2,8} = 4 * T_{2,2,2}. Rank(T_{2,2,2}) = 7 (Strassen). Rank(T_{2,2,8}) = 28.
    # T_{2,1,8} = 16 scalar mults. Rank(T_{2,1,8}) = 16.
    # Total Rank = 44.
    
    rank = 44
    # U shape: (n*m, rank) = (6, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (16, 44)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2x2 decomposition coefficients
    # U indices: i*m + j (i in 0..1, j in 0..1)
    # V indices: j*p + k (j in 0..1, k in 0..1)
    # W indices: k*n + i (k in 0..1, i in 0..1)
    
    # The 7 terms of Strassen's algorithm for 2x2 matrix mult
    # Each term is (u_coeff, v_coeff, w_coeff)
    # u_coeff: dict {(i,j): val}
    # v_coeff: dict {(j,k): val}
    # w_coeff: dict {(k,i): val}
    
    strassen_terms = [
        # 1. (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # 2. (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # 3. a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # 4. d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # 5. (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # 6. (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # 7. (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for u_c, v_c, w_c in strassen_terms:
            for (i, j), val in u_c.items():
                U[i_off + i, col] += val
            for (j, k), val in v_c.items():
                V[j_off * p + (j_off + j) * 0 + (k_off + k), col] += val # Careful with V index
                # V index is (j_global, k_global). 
                # j_global = j_off + j
                # k_global = k_off + k
                # Index in V = j_global * p + k_global
            # Wait, my V index calculation above was wrong in the comment.
            # Let's rewrite the loop correctly.
            
            # Reset and redo the inner loop logic properly
            col += 1
            # Actually, I should just apply the values directly.
            
    # Let's do it more simply without nested dicts for clarity and speed in the loop.
    # I will define the sparse updates directly.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    def apply_strassen(i_off, j_off, k_off):
        nonlocal col
        # 1
        U[i_off, col] += 1; U[i_off+1, col] += 1
        V[(j_off)*p + k_off, col] += 1; V[(j_off+1)*p + k_off+1, col] += 1
        W[k_off*n + i_off, col] += 1; W[(k_off+1)*n + i_off+1, col] += 1
        col += 1
        # 2
        U[i_off+1, col] += 1; U[i_off+1, col] += 1 # Wait, (c+d) is (1,0) and (1,1) in U?
        # U indices: i*m + j.
        # i_off=0, j_off=0.
        # Term 2: (c+d)e. c is (1,0), d is (1,1).
        # U[1*m+0] and U[1*m+1].
        # My previous manual check:
        # U[(i_off+1)*m + (j_off+0)] and U[(i_off+1)*m + (j_off+1)]
        # So:
        U[i_off+1, col] = 0 # Reset? No, we are adding.
        # Let's stick to the explicit index calculation from the seed code which was verified.
        
    # I will just reuse the logic from the seed code but write it cleanly.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # 1. (a+d)(e+h)
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 2. (c+d)e
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        # 3. a(f-h)
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 4. d(g-e)
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # 5. (a+b)h
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        # 6. (c-a)(e+f)
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 7. (b-d)(g+h)
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + (j_off+0), col] = 1
                V[(j_off+0)*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # Block 1: k=0..3
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 0) # j=0,1 k=0,1
    add_strassen_2x2x2(0, 0, 2) # j=0,1 k=2,3
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 0) # j=2 k=0..3

    # Block 2: k=4..7
    # m=2 part (j=0,1)
    add_strassen_2x2x2(0, 0, 4) # j=0,1 k=4,5
    add_strassen_2x2x2(0, 0, 6) # j=0,1 k=6,7
    # m=1 part (j=2)
    add_schoolbook_2x1x4(0, 2, 4) # j=2 k=4..7

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
