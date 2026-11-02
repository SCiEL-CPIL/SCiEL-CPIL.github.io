"""Matrix multiplication 4x4x7 - rank minimization via Kronecker/Strassen structure + refinement."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[i,j,k] with shape (n*m, m*p, p*n).
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all i,j,k."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Verify that sum_c U[:,c] V[:,c] W[:,c] == T."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _refine_adaptive(U, V, W, T, max_iters=300, tol=1e-9, base_lr=0.01):
    """Adaptive gradient descent with momentum and learning rate decay."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    mW = np.zeros_like(W)
    beta1 = 0.9
    lr = base_lr
    
    for it in range(max_iters):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = recon - T
        loss = np.sum(diff * diff)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        mU = beta1 * mU + (1 - beta1) * gU
        mV = beta1 * mV + (1 - beta1) * gV
        mW = beta1 * mW + (1 - beta1) * gW
        
        lr_t = base_lr / (1.0 + 0.01 * it)
        
        U -= lr_t * mU
        V -= lr_t * mV
        W -= lr_t * mW
    
    return best_U, best_V, best_W, best_loss

def _strassen_2x2_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2.
    Returns U, V, W each of shape (4, 7) with entries in {-1, 0, 1}."""
    # Strassen's 7 products for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # S1 = (a+d)(e+h)
    # S2 = (c+d)e
    # S3 = a(f-h)
    # S4 = d(g-e)
    # S5 = (a+b)h
    # S6 = (c-d)(f+g)
    # S7 = (b-d)(e+g)
    # C11 = S1+S4-S5+S7
    # C12 = S3+S5
    # C21 = S2+S4
    # C22 = S1-S2+S3+S6
    
    # U[i,j] for A factor, V[j,k] for B factor, W[k,i] for C factor
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # S1 = (a+d)(e+h): U=[1,0,0,1], V=[0,0,0,1], W=[1,0,0,1]
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # S2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # S3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    
    # S4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0], W=[1,0,0,0]
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[0,3]=1
    
    # S5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,0,1]
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[3,4]=1
    
    # S6 = (c-d)(f+g): U=[0,0,1,-1], V=[0,1,1,0], W=[0,0,0,1]
    U[2,5]=1; U[3,5]=-1
    V[1,5]=1; V[2,5]=1
    W[3,5]=1
    
    # S7 = (b-d)(e+g): U=[0,1,0,-1], V=[1,0,1,0], W=[1,0,0,0]
    U[1,6]=1; U[3,6]=-1
    V[0,6]=1; V[2,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_kronecker_init(n, m, p):
    """Build a Kronecker-structured initialization for 4x4x7.
    
    Split 4x4 into four 2x2 blocks. For the first 4 columns of B, use Strassen
    on each 2x2 block pair. For the remaining 3 columns, use schoolbook.
    
    This gives rank = 4*7 + 4*4*3 = 28 + 48 = 76 initially,
    but we can try to merge/share Strassen terms across blocks.
    
    Actually, the cleanest approach: treat the 4x4x7 multiplication as
    4 independent 2x2x7 multiplications (for the 4 row-blocks of A),
    each of which can be done with rank 7*4 = 28 (Strassen for the 2x2 part
    times 4 for the 7 columns... no that's not right either).
    
    Let me think more carefully. The 4x4x7 multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Split into 2x2 blocks:
    A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    where each block is 2x2 for A, 2x4 for B (since 7=4+3, let's use 4+3 split).
    
    Actually, let's use a simpler approach: 
    - For the first 4 columns of B (B[:, :4]), we have a 4x4x4 problem.
      This can be decomposed as 4 copies of 2x2x2 (one for each 2x2 block),
      each with rank 7, giving rank 4*7 = 28.
    - For the remaining 3 columns of B (B[:, 4:7]), we have a 4x4x3 problem.
      This can be done with schoolbook: rank 4*4*3 = 48.
    - Total: 28 + 48 = 76.
    
    But we can do better by sharing the Strassen structure.
    Let's just build it explicitly.
    """
    n, m, p = 4, 4, 7
    
    # Split B columns: first 4 use Strassen-structured, last 3 use schoolbook
    p_strassen = 4
    p_school = 3
    
    rank = 4 * 7 + 4 * 4 * p_school  # 28 + 48 = 76
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen part: 4 blocks of 2x2x2
    # Block (bi, bj) where bi in {0,1} (row block of A), bj in {0,1} (col block of B)
    # A block (bi,bj) is 2x2: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bj, bk) is 2x2: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block (bi, bk) is 2x2: rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    
    S_U, S_V, S_W = _strassen_2x2_2x2()
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for s in range(7):
                    # A factor: block (bi, bj), Strassen component s
                    for ii in range(2):
                        for jj in range(2):
                            a_idx = (2*bi + ii) * m + (2*bj + jj)
                            U[a_idx, col] = S_U[ii*2+jj, s]
                    
                    # B factor: block (bj, bk), Strassen component s
                    for jj in range(2):
                        for kk in range(2):
                            b_idx = (2*bj + jj) * p + (2*bk + kk)
                            V[b_idx, col] = S_V[jj*2+kk, s]
                    
                    # C factor: block (bi, bk), Strassen component s
                    for kk in range(2):
                        for ii in range(2):
                            c_idx = (2*bk + kk) * n + (2*bi + ii)
                            W[c_idx, col] = S_W[kk*2+ii, s]
                    
                    col += 1
    
    # Schoolbook part: 4x4x3 for columns 4-6 of B
    for i in range(n):
        for j in range(m):
            for k in range(p_strassen, p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W, rank

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _try_polish(U, V, W, T, max_iters=500, tol=1e-10):
    """High-precision polish pass."""
    return _refine_adaptive(U, V, W, T, max_iters=max_iters, tol=tol, base_lr=0.001)

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, schoolbook_rank = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Phase 1: Try Kronecker/Strassen structured initialization
    if time.time() < deadline - 60:
        U_k, V_k, W_k, rank_k = _build_kronecker_init(n, m, p)
        print(f"Kronecker init rank: {rank_k}")
        
        # Verify the structured init
        if _verify(U_k, V_k, W_k, T, tol=1e-6):
            best_U, best_V, best_W = U_k.copy(), V_k.copy(), W_k.copy()
            best_rank = rank_k
            print(f"Kronecker init verified at rank {rank_k}")
        
        # Try to refine and potentially reduce rank
        U_r, V_r, W_r, loss = _refine_adaptive(U_k, V_k, W_k, T, max_iters=200, tol=1e-8, base_lr=0.005)
        
        if loss < 1e-6:
            # Try to reduce rank by zeroing out columns with small contribution
            for target_rank in range(rank_k - 1, max(50, rank_k - 20), -1):
                if time.time() > deadline - 30:
                    break
                
                # Try truncating: keep the top 'target_rank' columns by norm
                U_trunc = U_r[:, :target_rank]
                V_trunc = V_r[:, :target_rank]
                W_trunc = W_r[:, :target_rank]
                
                # Refine the truncated version
                U_t, V_t, W_t, loss_t = _refine_adaptive(U_trunc, V_trunc, W_trunc, T, 
                                                          max_iters=100, tol=1e-8, base_lr=0.003)
                
                if loss_t < 1e-6 and _verify(U_t, V_t, W_t, T, tol=1e-6):
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank = target_rank
                    print(f"Reduced to rank {target_rank} with loss {loss_t:.2e}")
                    break
        
        # Polish if we have a good solution
        if best_rank < schoolbook_rank and time.time() < deadline - 20:
            U_p, V_p, W_p, loss_p = _try_polish(best_U, best_V, best_W, T, max_iters=200, tol=1e-9)
            if loss_p < 1e-8:
                best_U, best_V, best_W = U_p, V_p, W_p
    
    # Phase 2: Try random perturbations from the best solution to find lower rank
    if best_rank < schoolbook_rank and time.time() < deadline - 25:
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Add small random noise and try to find a lower-rank solution
            noise_scale = 0.1
            U_noisy = best_U + noise_scale * np.random.randn(*best_U.shape)
            V_noisy = best_V + noise_scale * np.random.randn(*best_V.shape)
            W_noisy = best_W + noise_scale * np.random.randn(*best_W.shape)
            
            # Try to reduce rank
            for target_rank in range(best_rank - 1, max(45, best_rank - 10), -1):
                if time.time() > deadline - 10:
                    break
                
                U_trunc = U_noisy[:, :target_rank]
                V_trunc = V_noisy[:, :target_rank]
                W_trunc = W_noisy[:, :target_rank]
                
                U_t, V_t, W_t, loss_t = _refine_adaptive(U_trunc, V_trunc, W_t, T,
                                                          max_iters=80, tol=1e-7, base_lr=0.002)
                
                if loss_t < 1e-6 and _verify(U_t, V_t, W_t, T, tol=1e-6):
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank = target_rank
                    print(f"Improved to rank {target_rank} via perturbation")
                    break
    
    # Final polish
    if time.time() < deadline - 5:
        U_f, V_f, W_f, loss_f = _try_polish(best_U, best_V, best_W, T, max_iters=100, tol=1e-9)
        if loss_f < 1e-7:
            best_U, best_V, best_W = U_f, V_f, W_f
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
