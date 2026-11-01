"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if is_valid:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def _build_rank11_block(base_col, block_start_k, U, V, W, n, m):
    """
    Add 11 terms for a 2x3 * 3x2 block starting at column block_start_k.
    
    For the 2x3 * 3x2 case with columns k0=block_start_k, k1=block_start_k+1:
    Terms (using local indices, then mapped to global):
    1. A[0,0] * B[0,k0]       -> U[0][base+0]=1, V[0*p+k0][base+0]=1, W[k0*n+0][base+0]=1
    2. A[0,1] * B[1,k0]       -> U[1][base+1]=1, V[1*p+k0][base+1]=1, W[k0*n+0][base+1]=1
    3. A[0,2] * B[2,k0]       -> U[2][base+2]=1, V[2*p+k0][base+2]=1, W[k0*n+0][base+2]=1
    4. A[1,0] * B[0,k0]       -> U[3][base+3]=1, V[0*p+k0][base+3]=1, W[k0*n+1][base+3]=1
    5. A[1,1] * B[1,k0]       -> U[4][base+4]=1, V[1*p+k0][base+4]=1, W[k0*n+1][base+4]=1
    6. A[1,2] * B[2,k0]       -> U[5][base+5]=1, V[2*p+k0][base+5]=1, W[k0*n+1][base+5]=1
    7. (A[0,0]+A[1,0]) * B[0,k1] -> U[0][base+6]=1, U[3][base+6]=1, V[0*p+k1][base+6]=1, W[k1*n+0][base+6]=1, W[k1*n+1][base+6]=1
    8. (A[0,1]+A[1,1]) * B[1,k1] -> U[1][base+7]=1, U[4][base+7]=1, V[1*p+k1][base+7]=1, W[k1*n+0][base+7]=1, W[k1*n+1][base+7]=1
    9. (A[0,2]+A[1,2]) * B[2,k1] -> U[2][base+8]=1, U[5][base+8]=1, V[2*p+k1][base+8]=1, W[k1*n+0][base+8]=1, W[k1*n+1][base+8]=1
    10. A[0,0] * (B[0,k1]-B[0,k0]) -> U[0][base+9]=1, V[0*p+k1][base+9]=1, V[0*p+k0][base+9]=-1, W[k1*n+0][base+9]=1
    11. A[0,1] * (B[1,k1]-B[1,k0]) -> U[1][base+10]=1, V[1*p+k1][base+10]=1, V[1*p+k0][base+10]=-1, W[k1*n+0][base+10]=1
    
    Verification:
    c00 (W[k0*n+0]) = term1 + term2 + term3 = A[0,0]B[0,k0] + A[0,1]B[1,k0] + A[0,2]B[2,k0] ✓
    c10 (W[k0*n+1]) = term4 + term5 + term6 = A[1,0]B[0,k0] + A[1,1]B[1,k0] + A[1,2]B[2,k0] ✓
    c01 (W[k1*n+0]) = term7 + term8 + term9 + term10 + term11
                     = (A[0,0]+A[1,0])B[0,k1] + (A[0,1]+A[1,1])B[1,k1] + (A[0,2]+A[1,2])B[2,k1]
                       + A[0,0](B[0,k1]-B[0,k0]) + A[0,1](B[1,k1]-B[1,k0])
                     = A[0,0]B[0,k1] + A[1,0]B[0,k1] + A[0,1]B[1,k1] + A[1,1]B[1,k1] + A[0,2]B[2,k1] + A[1,2]B[2,k1]
                       + A[0,0]B[0,k1] - A[0,0]B[0,k0] + A[0,1]B[1,k1] - A[0,1]B[1,k0]
                     = 2A[0,0]B[0,k1] + A[1,0]B[0,k1] + 2A[0,1]B[1,k1] + A[1,1]B[1,k1] + A[0,2]B[2,k1] + A[1,2]B[2,k1]
                       - A[0,0]B[0,k0] - A[0,1]B[1,k0]
    
    Wait, that's not right. Let me re-examine.
    
    Actually, looking at the W indices more carefully:
    - term7: W[k1*n+0] and W[k1*n+1] both get +1
    - term8: W[k1*n+0] and W[k1*n+1] both get +1  
    - term9: W[k1*n+0] and W[k1*n+1] both get +1
    - term10: W[k1*n+0] gets +1
    - term11: W[k1*n+0] gets +1
    
    So W[k1*n+0] (c01) gets contributions from terms 7,8,9,10,11
    W[k1*n+1] (c11) gets contributions from terms 7,8,9
    
    c01 = (A[0,0]+A[1,0])B[0,k1] + (A[0,1]+A[1,1])B[1,k1] + (A[0,2]+A[1,2])B[2,k1]
          + A[0,0](B[0,k1]-B[0,k0]) + A[0,1](B[1,k1]-B[1,k0])
        = A[0,0]B[0,k1] + A[1,0]B[0,k1] + A[0,1]B[1,k1] + A[1,1]B[1,k1] + A[0,2]B[2,k1] + A[1,2]B[2,k1]
          + A[0,0]B[0,k1] - A[0,0]B[0,k0] + A[0,1]B[1,k1] - A[0,1]B[1,k0]
        = 2A[0,0]B[0,k1] + A[1,0]B[0,k1] + 2A[0,1]B[1,k1] + A[1,1]B[1,k1] + A[0,2]B[2,k1] + A[1,2]B[2,k1]
          - A[0,0]B[0,k0] - A[0,1]B[1,k0]
    
    This is NOT equal to A[0,0]B[0,k1] + A[0,1]B[1,k1] + A[0,2]B[2,k1].
    
    Let me reconsider the construction. The issue is that terms 7,8,9 contribute to BOTH c01 and c11.
    
    Let me redesign:
    c11 = A[1,0]B[0,k1] + A[1,1]B[1,k1] + A[1,2]B[2,k1]
    c01 = A[0,0]B[0,k1] + A[0,1]B[1,k1] + A[0,2]B[2,k1]
    
    If I define:
    term7 = (A[0,0]+A[1,0]) * B[0,k1]  -> contributes to both c01 and c11
    term8 = (A[0,1]+A[1,1]) * B[1,k1]  -> contributes to both c01 and c11
    term9 = (A[0,2]+A[1,2]) * B[2,k1]  -> contributes to both c01 and c11
    
    Then c01 + c11 = term7 + term8 + term9
    
    I need to recover c01 and c11 separately.
    c01 = c01
    c11 = (term7+term8+term9) - c01
    
    So I need 3 more terms for c01, giving 6+3+3=12.
    
    To get 11, I need to share one term. Let me try:
    term10 = A[0,0] * B[0,k1]  (direct term for c01)
    term11 = A[0,1] * B[1,k1]  (direct term for c01)
    
    Then c01 = term10 + term11 + A[0,2]B[2,k1]
    But I don't have A[0,2]B[2,k1] as a separate term.
    
    Alternative: 
    term10 = A[0,0] * (B[0,k1] - B[0,k0])
    term11 = A[0,1] * (B[1,k1] - B[1,k0])
    
    c01 = term1 + term2 + term3 + term10 + term11
        = A[0,0]B[0,k0] + A[0,1]B[1,k0] + A[0,2]B[2,k0] + A[0,0](B[0,k1]-B[0,k0]) + A[0,1](B[1,k1]-B[1,k0])
        = A[0,0]B[0,k1] + A[0,1]B[1,k1] + A[0,2]B[2,k0]
    
    That's not right either - I need A[0,2]B[2,k1], not A[0,2]B[2,k0].
    
    Let me try a completely different approach. What if I use:
    term10 = A[0,2] * B[2,k1]
    
    Then c01 = term10 + ... but I still need A[0,0]B[0,k1] and A[0,1]B[1,k1].
    
    It seems 11 is not achievable with this simple sharing strategy.
    
    Let me just return the rank-48 solution for now, as the rank-11 construction is not working.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Use rank-48 schoolbook solution
    rank = n * m * p  # 48
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
    
    assert _verify(U, V, W, n, m, p, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
