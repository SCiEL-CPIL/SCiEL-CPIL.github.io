import numpy as np
import time

def _build_rank44():
    """Build rank-44 decomposition: 4 blocks of 2x3*3x2, each rank 11."""
    n, m, p = 2, 3, 8
    rank_per_block = 11
    num_blocks = p // 2
    total_rank = rank_per_block * num_blocks
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        k0, k1 = 2*blk, 2*blk+1
        
        # For this block, B columns are k0, k1 (rows 0,1,2 of B)
        # B indices: row0: k0,k1 -> idx k0, k1
        #             row1: k0,k1 -> idx p+k0, p+k1
        #             row2: k0,k1 -> idx 2p+k0, 2p+k1
        # C indices: C[0,k0]=4blk, C[1,k0]=4blk+1, C[0,k1]=4blk+2, C[1,k1]=4blk+3
        
        # Use correct Strassen for the 2x2 core (A[:,0:2] * B[0:2,:])
        # p1 = (A00+A11)(B00+B11)
        U[0, base+0] = 1;  U[4, base+0] = 1
        V[k0, base+0] = 1; V[p+k1, base+0] = 1
        W[4*blk+0, base+0] = 1; W[4*blk+3, base+0] = 1
        
        # p2 = (A10+A11)B00
        U[3, base+1] = 1;  U[4, base+1] = 1
        V[k0, base+1] = 1
        W[4*blk+1, base+1] = 1; W[4*blk+3, base+1] = -1
        
        # p3 = A00(B01-B11)
        U[0, base+2] = 1
        V[k1, base+2] = 1; V[p+k1, base+2] = -1
        W[4*blk+2, base+2] = 1; W[4*blk+3, base+2] = 1
        
        # p4 = A11(B10-B00)
        U[4, base+3] = 1
        V[p+k0, base+3] = 1; V[k0, base+3] = -1
        W[4*blk+0, base+3] = 1; W[4*blk+1, base+3] = 1
        
        # p5 = (A00+A01)B11
        U[0, base+4] = 1;  U[1, base+4] = 1
        V[p+k1, base+4] = 1
        W[4*blk+0, base+4] = -1; W[4*blk+2, base+4] = 1
        
        # p6 = (A10-A00)(B00+B01)
        U[3, base+5] = 1;  U[0, base+5] = -1
        V[k0, base+5] = 1; V[k1, base+5] = 1
        W[4*blk+3, base+5] = 1
        
        # p7 = (A01-A11)(B10+B11)
        U[1, base+6] = 1;  U[4, base+6] = -1
        V[p+k0, base+6] = 1; V[p+k1, base+6] = 1
        W[4*blk+0, base+6] = 1
        
        # Extra: A[0,2]*B[2,k0], A[0,2]*B[2,k1], A[1,2]*B[2,k0], A[1,2]*B[2,k1]
        U[2, base+7] = 1
        V[2*p+k0, base+7] = 1
        W[4*blk+0, base+7] = 1
        
        U[2, base+8] = 1
        V[2*p+k1, base+8] = 1
        W[4*blk+2, base+8] = 1
        
        U[5, base+9] = 1
        V[2*p+k0, base+9] = 1
        W[4*blk+1, base+9] = 1
        
        U[5, base+10] = 1
        V[2*p+k1, base+10] = 1
        W[4*blk+3, base+10] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n=2, m=3, p=8, tol=1e-10):
    """Verify decomposition numerically."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_flat = C_expected.flatten()  # index k*n+i
    
    U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
    
    rank = U_arr.shape[1]
    C_decomp = np.zeros(p * n)
    for r in range(rank):
        u_mat = U_arr[:, r].reshape(n, m)
        v_mat = V_arr[:, r].reshape(m, p)
        c_contrib = (u_mat @ v_mat).T.flatten()
        C_decomp += W_arr[:, r] * c_contrib
    
    if not np.allclose(C_decomp, C_flat, atol=tol):
        return False, np.max(np.abs(C_decomp - C_flat))
    return True, 0.0

def _try_rank_reduction(U44, V44, W44, max_time=100.0):
    """Check for redundant proportional terms."""
    start = time.time()
    rank = U44.shape[1]
    for r1 in range(rank):
        if time.time() - start > max_time - 10:
            break
        for r2 in range(r1+1, rank):
            u1, u2 = U44[:, r1], U44[:, r2]
            v1, v2 = V44[:, r1], V44[:, r2]
            w1, w2 = W44[:, r1], W44[:, r2]
            nz_u1, nz_u2 = np.nonzero(u1)[0], np.nonzero(u2)[0]
            if len(nz_u1) == 0 or len(nz_u2) == 0 or set(nz_u1) != set(nz_u2):
                continue
            ratios = u2[nz_u1] / u1[nz_u1]
            if not np.allclose(ratios, ratios[0]):
                continue
            alpha = ratios[0]
            nz_v1, nz_v2 = np.nonzero(v1)[0], np.nonzero(v2)[0]
            if set(nz_v1) != set(nz_v2):
                continue
            rv = v2[nz_v1] / v1[nz_v1]
            if not np.allclose(rv, rv[0]) or not np.allclose(rv[0], alpha):
                continue
            nz_w1, nz_w2 = np.nonzero(w1)[0], np.nonzero(w2)[0]
            if set(nz_w1) != set(nz_w2):
                continue
            rw = w2[nz_w1] / w1[nz_w1]
            if not np.allclose(rw, rw[0]) or not np.allclose(rw[0], alpha):
                continue
            mask = np.ones(rank, dtype=bool)
            mask[r2] = False
            U_new, V_new, W_new = U44[:, mask], V44[:, mask], W44[:, mask]
            if _verify_decomp(U_new, V_new, W_new)[0]:
                return U_new, V_new, W_new
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    U44, V44, W44 = _build_rank44()
    best_U, best_V, best_W = U44, V44, W44
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        result = _try_rank_reduction(U44, V44, W44, max_time=remaining_time - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomp(U_new, V_new, W_new)[0] and U_new.shape[1] < U44.shape[1]:
                best_U, best_V, best_W = U_new, V_new, W_new
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
