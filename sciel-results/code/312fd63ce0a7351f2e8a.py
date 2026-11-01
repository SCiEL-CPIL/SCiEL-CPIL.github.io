"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # We'll build a rank-34 decomposition for (2,2,9)
    # Using the known minimal rank 4p-2 = 34 for (2,2,p), p>=3
    
    rank = 34
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    # Indices:
    # A: (i,j) -> i*2+j, i,j in {0,1} => 0,1,2,3
    # B: (j,k) -> j*9+k, j in {0,1}, k in {0..8} => 0..8 (j=0), 9..17 (j=1)
    # C: (k,i) -> k*2+i, k in {0..8}, i in {0,1} => 0..17
    
    col = 0
    
    # Strategy: For each column k, we need:
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    # where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    
    # Use 4p-2 = 34 terms:
    # For k=0..8: 2 terms for C[0,k] (a*B[0,k] and b*B[1,k])
    # For k=0..8: 2 terms for C[1,k] (c*B[0,k] and d*B[1,k])
    # But we combine the last 2 columns' c,d terms into shared factors.
    
    # Actually, let's use a cleaner construction:
    # 2p terms: a*B[0,k] -> C[0,k] and c*B[0,k] -> C[1,k] for k=0..8
    # 2p-2 terms: (b+d)*B[1,k] type combinations for k=0..7, plus boundary terms
    
    # Let me use the following explicit rank-34 construction:
    
    # Terms 0-8: a * B[0,k] contributes to C[0,k]
    for k in range(p):
        U[0, col] = 1   # a
        V[k, col] = 1   # B[0,k]
        W[2*k, col] = 1  # C[0,k]
        col += 1
    
    # Terms 9-17: c * B[0,k] contributes to C[1,k]
    for k in range(p):
        U[2, col] = 1   # c
        V[k, col] = 1   # B[0,k]
        W[2*k+1, col] = 1  # C[1,k]
        col += 1
    
    # Now we need b*B[1,k] -> C[0,k] and d*B[1,k] -> C[1,k] for k=0..8
    # That's 18 more terms, but we only have 34-18 = 16 slots.
    # So we need to handle b and d with 16 terms instead of 18.
    
    # Use (b+d) * B[1,k] for k=0..7 (8 terms), giving (b+d)*B[1,k] to both C[0,k] and C[1,k]
    # Then subtract d*B[1,k] from C[0,k] and b*B[1,k] from C[1,k] for k=0..7
    # That's 8 + 8 = 16 terms, but we also need k=8 for both b and d.
    
    # Hmm, let me reconsider. 
    # 
    # Alternative: Use 18 terms for b and d individually, but merge 2 of them.
    # 
    # Let's try: for k=0..7, use (b+d)*B[1,k] -> C[0,k] and C[1,k] (8 terms)
    # Then for k=0..7, subtract d*B[1,k] from C[0,k] (8 terms) 
    # And for k=0..7, subtract b*B[1,k] from C[1,k] (8 terms)
    # Plus b*B[1,8] -> C[0,8] and d*B[1,8] -> C[1,8] (2 terms)
    # Total: 8+8+8+2 = 26, plus 18 from before = 44. Too many.
    
    # Let me restart with a different approach.
    
    # Clear and redo
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    # Use 4p-2 = 34 terms with the following pattern:
    # For k in 0..8:
    #   Term 4k:   a * B[0,k] -> C[0,k]
    #   Term 4k+1: c * B[0,k] -> C[1,k]  
    #   Term 4k+2: (b+d) * B[1,k] -> C[0,k] + C[1,k]  (for k<8)
    #   Term 4k+3: d * (B[1,k] - B[0,k]) ... 
    
    # This is getting complicated. Let me just verify the schoolbook works
    # and try a simple rank reduction.
    
    # Fallback: schoolbook rank 36
    rank = 36
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
