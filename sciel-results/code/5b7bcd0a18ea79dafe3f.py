import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    def build_rank44():
        rank_per_block = 11
        num_blocks = p // 2
        total_rank = rank_per_block * num_blocks
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for blk in range(num_blocks):
            base = blk * rank_per_block
            k0 = 2 * blk
            k1 = 2 * blk + 1
            
            # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
            # B indices: B[j,k] = j*8+k
            # C indices: C[i,k] = k*2+i (trace convention)
            
            # Strassen for 2x2 (A cols 0,1; B rows 0,1; C rows 0,1, cols k0,k1)
            # m1 = (A00+A11)(B00+B11) -> C00+=, C11+=
            U[0, base+0] = 1  # A[0,0]
            U[4, base+0] = 1  # A[1,1]
            V[0*8+k0, base+0] = 1  # B[0,k0]
            V[1*8+k1, base+0] = 1  # B[1,k1]
            W[k0*2+0, base+0] = 1  # C[0,k0]
            W[k1*2+1, base+0] = 1  # C[1,k1]
            
            # m2 = A10(B00+B10) -> C10+=, C11-=
            U[3, base+1] = 1  # A[1,0]
            U[4, base+1] = 1  # A[1,1]
            V[0*8+k0, base+1] = 1  # B[0,k0]
            W[k0*2+1, base+1] = 1  # C[1,k0]
            W[k1*2+1, base+1] = -1  # C[1,k1]
            
            # m3 = A00(B10-B11) -> C10+=, C11+=
            U[0, base+2] = 1  # A[0,0]
            V[1*8+k0, base+2] = 1  # B[1,k0]
            V[1*8+k1, base+2] = -1  # B[1,k1]
            W[k0*2+1, base+2] = 1  # C[1,k0]
            W[k1*2+1, base+2] = 1  # C[1,k1]
            
            # m4 = A11(B01-B00) -> C00+=, C01+=
            U[4, base+3] = 1  # A[1,1]
            V[0*8+k1, base+3] = 1  # B[0,k1]
            V[0*8+k0, base+3] = -1  # B[0,k0]
            W[k0*2+0, base+3] = 1  # C[0,k0]
            W[k0*2+1, base+3] = 1  # C[0,k1]
            
            # m5 = (A01+A10)(B00+B01) -> C00-=, C01+=
            U[1, base+4] = 1  # A[0,1]
            U[3, base+4] = 1  # A[1,0]
            V[0*8+k0, base+4] = 1  # B[0,k0]
            V[0*8+k1, base+4] = 1  # B[0,k1]
            W[k0*2+0, base+4] = -1  # C[0,k0]
            W[k0*2+1, base+4] = 1  # C[0,k1]
            
            # m6 = (A10+A11)(B10+B11) -> C11+=
            U[3, base+5] = 1  # A[1,0]
            U[4, base+5] = 1  # A[1,1]
            V[1*8+k0, base+5] = 1  # B[1,k0]
            V[1*8+k1, base+5] = 1  # B[1,k1]
            W[k1*2+1, base+5] = 1  # C[1,k1]
            
            # m7 = (A00+A01)(B11+B10) -> C00+=
            U[0, base+6] = 1  # A[0,0]
            U[1, base+6] = 1  # A[0,1]
            V[1*8+k1, base+6] = 1  # B[1,k1]
            V[1*8+k0, base+6] = 1  # B[1,k0]
            W[k0*2+0, base+6] = 1  # C[0,k0]
            
            # Rank-1 terms for A col 2
            U[2, base+7] = 1   # A[0,2]
            V[2*8+k0, base+7] = 1  # B[2,k0]
            W[k0*2+0, base+7] = 1  # C[0,k0]
            
            U[2, base+8] = 1   # A[0,2]
            V[2*8+k1, base+8] = 1  # B[2,k1]
            W[k1*2+0, base+8] = 1  # C[0,k1]
            
            U[5, base+9] = 1   # A[1,2]
            V[2*8+k0, base+9] = 1  # B[2,k0]
            W[k0*2+1, base+9] = 1  # C[1,k0]
            
            U[5, base+10] = 1  # A[1,2]
            V[2*8+k1, base+10] = 1  # B[2,k1]
            W[k1*2+1, base+10] = 1  # C[1,k1]
        
        return U, V, W, total_rank
    
    def verify_full(U, V, W):
        rank = U.shape[1]
        for a in range(n*m):
            i_a, j_a = divmod(a, 3)
            for b in range(m*p):
                j_b, k_b = divmod(b, 8)
                for c in range(p*n):
                    k_c, i_c = divmod(c, 2)
                    expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    U44, V44, W44, rank44 = build_rank44()
    valid44 = verify_full(U44, V44, W44)
    
    if not valid44:
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
    
    if time.time() > deadline:
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
    
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}
