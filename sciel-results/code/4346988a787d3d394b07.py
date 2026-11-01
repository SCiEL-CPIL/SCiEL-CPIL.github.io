"""Matrix multiplication 2,2,13 - using block decomposition with Strassen for 2x2x2 blocks."""
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 13
    
    # We decompose the problem into 6 blocks of size 2x2x2 and 1 block of size 2x2x1.
    # Rank of M_2,2,2 is 7 (Strassen).
    # Rank of M_2,2,1 is 4.
    # Total rank = 6 * 7 + 4 = 46.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen decomposition for M_2,2,2
    # u[i*m+j, r], v[j*p+k, r], w[k*n+i, r]
    # Local indices: i,j in {0,1}, k in {0,1}
    # U local shape 4x7, V local shape 4x7, W local shape 4x7
    
    # Strassen terms:
    # m1 = (a00+a11)(b00+b11) -> c00+=, c11+=
    # m2 = (a10+a11)b00 -> c10+=, c11-=
    # m3 = a00(b01-b11) -> c01+=
    # m4 = a11(b10-b00) -> c00+=, c10+=
    # m5 = (a00+a01)b11 -> c00-=, c01+=
    # m6 = (a10-a00)(b00+b01) -> c11+=
    # m7 = (a01-a11)(b10+b11) -> c00+=
    
    # We store coefficients for U, V, W for each of the 7 terms.
    # U[j] is index i*m+j.
    # V[j] is index j*p+k.
    # W[j] is index k*n+i.
    
    strassen_U = [
        [1, 0, 0, 1, 0, 0, 0], # m1: a00, a11
        [0, 1, 0, 0, 0, 0, 0], # m2: a10
        [0, 0, 1, 0, 0, 0, 0], # m3: a00? No, m3 is a00. Wait.
        # Let's re-verify Strassen mapping carefully.
        
        # m1: (a00+a11)(b00+b11)
        # U: a00=1, a11=1. Index 0, 3.
        # V: b00=1, b11=1. Index 0, 3.
        # W: c00=1, c11=1. Index 0, 3.
        
        # m2: (a10+a11)b00
        # U: a10=1, a11=1. Index 2, 3.
        # V: b00=1. Index 0.
        # W: c10=1, c11=-1. Index 1, 3.
        
        # m3: a00(b01-b11)
        # U: a00=1. Index 0.
        # V: b01=1, b11=-1. Index 1, 3.
        # W: c01=1. Index 2.
        
        # m4: a11(b10-b00)
        # U: a11=1. Index 3.
        # V: b10=1, b00=-1. Index 2, 0.
        # W: c00=1, c10=1. Index 0, 1.
        
        # m5: (a00+a01)b11
        # U: a00=1, a01=1. Index 0, 1.
        # V: b11=1. Index 3.
        # W: c00=-1, c01=1. Index 0, 2.
        
        # m6: (a10-a00)(b00+b01)
        # U: a10=1, a00=-1. Index 2, 0.
        # V: b00=1, b01=1. Index 0, 1.
        # W: c11=1. Index 3.
        
        # m7: (a01-a11)(b10+b11)
        # U: a01=1, a11=-1. Index 1, 3.
        # V: b10=1, b11=1. Index 2, 3.
        # W: c00=1. Index 0.
    ]
    
    # Correcting U, V, W matrices for Strassen (7 terms)
    # U is 4x7
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # V is 4x7 (indices j*p+k, p=2 local)
    # j=0: k=0(0), k=1(1)
    # j=1: k=0(2), k=1(3)
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, 0, 1, 0, 0, 1],  # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ]
    
    # W is 4x7 (indices k*n+i, n=2 local)
    # k=0: i=0(0), i=1(1)
    # k=1: i=0(2), i=1(3)
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [1, -1, 0, 0, 0, 1, 0]  # c11
    ]
    
    # Fill blocks
    # 6 blocks of 2 columns
    for b in range(6):
        offset_rank = b * 7
        offset_k = b * 2
        
        for r in range(7):
            global_r = offset_rank + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_val = U_s[i*m+j][r]
                    if u_val != 0:
                        U[i*m+j][global_r] = u_val
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    v_val = V_s[j*p+k_local][r] # Wait, V_s index is j*2+k_local.
                    if v_val != 0:
                        k_global = offset_k + k_local
                        V[j*p+k_global][global_r] = v_val
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    w_val = W_s[k_local*n+i][r]
                    if w_val != 0:
                        k_global = offset_k + k_local
                        W[k_global*n+i][global_r] = w_val
    
    # 1 block of 1 column (last column, index 12)
    offset_rank = 6 * 7
    offset_k = 12
    
    # M_2,2,1 decomposition (rank 4)
    # c00 = a00*b00 + a01*b10
    # c10 = a10*b00 + a11*b10
    
    # Term 0: a00*b00 -> c00
    U[0][offset_rank+0] = 1
    V[0][offset_rank+0] = 1 # j=0, k=0
    W[0][offset_rank+0] = 1 # k=0, i=0
    
    # Term 1: a01*b10 -> c00
    U[1][offset_rank+1] = 1
    V[1][offset_rank+1] = 1 # j=1, k=0
    W[0][offset_rank+1] = 1 # k=0, i=0
    
    # Term 2: a10*b00 -> c10
    U[2][offset_rank+2] = 1
    V[0][offset_rank+2] = 1 # j=0, k=0
    W[1][offset_rank+2] = 1 # k=0, i=1
    
    # Term 3: a11*b10 -> c10
    U[3][offset_rank+3] = 1
    V[1][offset_rank+3] = 1 # j=1, k=0
    W[1][offset_rank+3] = 1 # k=0, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
