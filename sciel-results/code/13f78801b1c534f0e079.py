import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 120.0

    n, m, p = 2, 3, 8
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        k0 = 2 * blk      # first column of this block
        k1 = 2 * blk + 1  # second column of this block
        
        # Strassen for 2x2 core: A[:,0:2] * B[0:2, :]
        # m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, base+0] = 1; U[4, base+0] = 1
        V[k0, base+0] = 1; V[p + k1, base+0] = 1
        W[4*blk + 0, base+0] = 1; W[4*blk + 3, base+0] = 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, base+1] = 1; U[4, base+1] = 1
        V[k0, base+1] = 1
        W[4*blk + 1, base+1] = 1; W[4*blk + 3, base+1] = -1
        
        # m3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, base+2] = 1
        V[k1, base+2] = 1; V[p + k1, base+2] = -1
        W[4*blk + 2, base+2] = 1; W[4*blk + 3, base+2] = 1
        
        # m4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4, base+3] = 1
        V[p + k0, base+3] = 1; V[k0, base+3] = -1
        W[4*blk + 0, base+3] = 1; W[4*blk + 1, base+3] = 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, base+4] = 1; U[1, base+4] = 1
        V[p + k1, base+4] = 1
        W[4*blk + 0, base+4] = -1; W[4*blk + 2, base+4] = 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3, base+5] = 1; U[0, base+5] = -1
        V[k0, base+5] = 1; V[k1, base+5] = 1
        W[4*blk + 3, base+5] = 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, base+6] = 1; U[4, base+6] = -1
        V[p + k0, base+6] = 1; V[p + k1, base+6] = 1
        W[4*blk + 0, base+6] = 1
        
        # Extra: A[i,2] * B[2,k] for i in {0,1}, k in {k0, k1}
        # C[0,k0] += A[0,2]*B[2,k0]
        U[2, base+7] = 1
        V[2*p + k0, base+7] = 1
        W[4*blk + 0, base+7] = 1
        
        # C[0,k1] += A[0,2]*B[2,k1]
        U[2, base+8] = 1
        V[2*p + k1, base+8] = 1
        W[4*blk + 2, base+8] = 1
        
        # C[1,k0] += A[1,2]*B[2,k0]
        U[5, base+9] = 1
        V[2*p + k0, base+9] = 1
        W[4*blk + 1, base+9] = 1
        
        # C[1,k1] += A[1,2]*B[2,k1]
        U[5, base+10] = 1
        V[2*p + k1, base+10] = 1
        W[4*blk + 3, base+10] = 1
    
    if time.time() > _deadline - 5.0:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
