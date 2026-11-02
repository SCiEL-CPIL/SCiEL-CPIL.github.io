import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, atol=1e-8):
    """Verify that U, V, W correctly decompose the n×m×p matrix multiplication tensor."""
    r = U.shape[1]
    D = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
    return np.allclose(D, 1.0, atol=atol)

def _strassen_2x2x2():
    """Return Strassen's rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7) with indices:
    U[i*2+j, c], V[i*2+j, c], W[i*2+j, c] where i=row of A, j=col of A (for U)
    For V: i=row of B, j=col of B
    For W: i=row of C, j=col of C
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1; W[3, 1] = 1  # C10, C11
    
    # M3: A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _combine_split_m(U_blocks, V_blocks, W_blocks):
    """Combine decompositions when splitting the middle dimension m.
    U_blocks: list of (n*m_i, r_i) arrays for A sub-blocks
    V_blocks: list of (m_i*p, r_i) arrays for B sub-blocks  
    W_blocks: list of (p*n, r_i) arrays for C
    Returns combined (n*m, total_r), (m*p, total_r), (p*n, total_r)
    """
    n = U_blocks[0].shape[0] // U_blocks[0].shape[1] if False else None
    # We need to know dimensions; pass them explicitly
    pass

def _block_decompose(n, m, p):
    """Recursively decompose n×m×p matrix multiplication using block structure.
    Returns U (n*m, r), V (m*p, r), W (p*n, r) with integer entries.
    """
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    
    if n == 1 and m == 1:
        # (1x1)@(1xp): rank p
        U = np.ones((1, p))
        V = np.zeros((p, p))
        W = np.zeros((p, p))
        for k in range(p):
            V[k, k] = 1
            W[k, k] = 1
        return U, V, W
    
    if m == 1 and p == 1:
        # (nx1)@(1x1): rank n
        U = np.zeros((n, n))
        V = np.ones((1, n))
        W = np.zeros((n, n))
        for i in range(n):
            U[i, i] = 1
            W[i, i] = 1
        return U, V, W
    
    if n == 1:
        # (1xm)@(mxp): rank m*p... but can we do better?
        # Actually for (1xm)@(mxp), each output C[0,k] = sum_j A[0,j]*B[j,k]
        # This is a set of p dot products of length m.
        # The tensor has rank at most m*p (schoolbook).
        # For m=2, p=2: rank 4 (schoolbook), but actually it's just 4 independent bilinear forms.
        # The rank is exactly m*p for generic case since each (j,k) pair gives independent term.
        # Wait no - A[0,j]*B[j,k] for all j,k. These are m*p terms but they share A[0,j] factors.
        # Actually the tensor T[0,j,k] = 1 for all j,k. This is the "all ones" tensor of size 1×m×p.
        # The rank of the all-ones tensor [1]^{m×p} is... it's the outer product structure.
        # T[j,k] = 1 = 1*1, so it's rank 1! No wait, that's for a single entry.
        # The tensor T with T[0,j,k]=1 for all j,k can be written as sum_c U[0,c]*V[j,c]*W[k,c]
        # We need: for all j,k: sum_c U[0,c]*V[j,c]*W[k,c] = 1
        # This means (U[0,:] * V) @ W^T = ones(m,p)
        # If we set U[0,c]=1 for all c, then we need V@W^T = ones(m,p)
        # V is (m,r), W is (p,r). V@W^T is (m,p).
        # The rank of the all-ones matrix (m×p) is 1.
        # So we need V@W^T = ones(m,p), which has matrix rank 1.
        # So r=1 suffices: V = ones(m,1), W = ones(p,1), U = [1].
        # Let's verify: sum_c=1 U[0,0]*V[j,0]*W[k,0] = 1*1*1 = 1. Yes!
        U = np.ones((1, 1))
        V = np.ones((m, 1))
        W = np.ones((p, 1))
        return U, V, W
    
    if m == 1:
        # (nx1)@(1xp): rank n*p... 
        # T[i,0,k] = 1 for all i,k. Same argument: rank 1.
        U = np.ones((n, 1))
        V = np.ones((1, 1))
        W = np.ones((p, 1))
        return U, V, W
    
    if p == 1:
        # (nxm)@(mx1): rank n*m...
        # T[i,j,0] = 1 for all i,j. Rank 1.
        U = np.ones((n*m, 1))
        V = np.ones((m, 1))
        W = np.ones((n, 1))
        return U, V, W
    
    # Try to use Strassen when all dimensions are even
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        # Split into 2x2x2 blocks
        # A is n×m, B is m×p, C is n×p
        # Split: A into 4 blocks (n/2 × m/2), B into 4 blocks (m/2 × p/2), C into 4 blocks (n/2 × p/2)
        # Use Strassen on each 2x2 block of blocks
        # Actually, we can apply Strassen at the block level.
        # Let's split n=n1+n2, m=m1+m2, p=p1+p2 where n1=n//2, etc.
        n1, n2 = n // 2, n - n // 2
        m1, m2 = m // 2, m - m // 2
        p1, p2 = p // 2, p - p // 2
        
        # We need decompositions for:
        # (n1×m1)@(m1×p1), (n1×m1)@(m1×p2), (n1×m2)@(m2×p1), (n1×m2)@(m2×p2)
        # (n2×m1)@(m1×p1), (n2×m1)@(m1×p2), (n2×m2)@(m2×p1), (n2×m2)@(m2×p2)
        # And Strassen combines them.
        
        # But this gets complex. Let me just handle the specific case n=2,m=2,p=2 directly.
        if n == 2 and m == 2 and p == 2:
            return _strassen_2x2x2()
    
    # For n=3, m=4, p=6: try specific block decompositions
    if n == 3 and m == 4 and p == 6:
        # Split m=4 into 2+2
        # C = A[:,0:2]@B[0:2,:] + A[:,2:4]@B[2:4,:]
        # Each term is (3×2)@(2×6)
        U1, V1, W1 = _block_decompose(3, 2, 6)
        U2, V2, W2 = _block_decompose(3, 2, 6)
        
        # Combine: U is (12, r1+r2), V is (24, r1+r2), W is (18, r1+r2)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 + r2
        
        U = np.zeros((12, r))
        V = np.zeros((24, r))
        W = np.zeros((18, r))
        
        # First block: A[:,0:2] is rows 0,2,4,6,8,10 of U (i*2+j for i in 0..2, j in 0..1)
        # B[0:2,:] is rows 0..11 of V (j*6+k for j in 0..1, k in 0..5)
        for c in range(r1):
            for i in range(3):
                for j in range(2):
                    U[i*4+j, c] = U1[i*2+j, c]
            for j in range(2):
                for k in range(6):
                    V[j*6+k, c] = V1[j*6+k, c]
            for k in range(6):
                for i in range(3):
                    W[k*3+i, c] = W1[k*3+i, c]
        
        # Second block: A[:,2:4] is rows 1,3,5,7,9,11 of U (i*2+1+j for i in 0..2, j in 0..1)
        # B[2:4,:] is rows 12..23 of V (j*6+k for j in 0..1, k in 0..5) offset by 12
        for c in range(r2):
            for i in range(3):
                for j in range(2):
                    U[i*4+(2+j), r1+c] = U2[i*2+j, c]
            for j in range(2):
                for k in range(6):
                    V[(2+j)*6+k, r1+c] = V2[j*6+k, c]
            for k in range(6):
                for i in range(3):
                    W[k*3+i, r1+c] = W2[k*3+i, c]
        
        return U, V, W
    
    if n == 3 and m == 2 and p == 6:
        # (3×2)@(2×6): split p=6 into 2+2+2
        # C = A@B[:,0:2] + A@B[:,2:4] + A@B[:,4:6]
        # Each term is (3×2)@(2×2)
        U1, V1, W1 = _block_decompose(3, 2, 2)
        U2, V2, W2 = _block_decompose(3, 2, 2)
        U3, V3, W3 = _block_decompose(3, 2, 2)
        
        r1, r2, r3 = U1.shape[1], U2.shape[1], U3.shape[1]
        r = r1 + r2 + r3
        
        U = np.zeros((6, r))
        V = np.zeros((12, r))
        W = np.zeros((18, r))
        
        # First block: B[:,0:2]
        for c in range(r1):
            U[:, c] = U1[:, c]
            for j in range(2):
                for k in range(2):
                    V[j*6+k, c] = V1[j*2+k, c]
            for k in range(2):
                for i in range(3):
                    W[k*3+i, c] = W1[k*3+i, c]
        
        # Second block: B[:,2:4]
        for c in range(r2):
            U[:, r1+c] = U2[:, c]
            for j in range(2):
                for k in range(2):
                    V[j*6+(2+k), r1+c] = V2[j*2+k, c]
            for k in range(2):
                for i in range(3):
                    W[(2+k)*3+i, r1+c] = W2[k*3+i, c]
        
        # Third block: B[:,4:6]
        for c in range(r3):
            U[:, r1+r2+c] = U3[:, c]
            for j in range(2):
                for k in range(2):
                    V[j*6+(4+k), r1+r2+c] = V3[j*2+k, c]
            for k in range(2):
                for i in range(3):
                    W[(4+k)*3+i, r1+r2+c] = W3[k*3+i, c]
        
        return U, V, W
    
    if n == 3 and m == 2 and p == 2:
        # (3×2)@(2×2): split n=3 into 1+2
        # C_top = A_top @ B (1×2)
        # C_bot = A_bot @ B (2×2)
        U1, V1, W1 = _block_decompose(1, 2, 2)
        U2, V2, W2 = _block_decompose(2, 2, 2)
        
        r1, r2 = U1.shape[1], U2.shape[1]
        r = r1 + r2
        
        U = np.zeros((6, r))
        V = np.zeros((4, r))
        W = np.zeros((6, r))
        
        # First block: A_top (1×2)
        for c in range(r1):
            U[0:2, c] = U1[0:2, c]  # U1 is (2, r1) for (1×2)
            V[:, c] = V1[:, c]
            W[0:2, c] = W1[0:2, c]  # W1 is (2, r1) for (1×2) output is 1×2, so W1 shape (2, r1)
        
        # Second block: A_bot (2×2)
        for c in range(r2):
            U[2:6, r1+c] = U2[:, c]  # U2 is (4, r2) for (2×2)
            V[:, r1+c] = V2[:, c]
            W[2:6, r1+c] = W2[:, c]  # W2 is (4, r2) for (2×2) output is 2×2
        
        return U, V, W
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    if n == 1 and m == 2 and p == 2:
        # (1×2)@(2×2): rank 1 as shown above
        U = np.ones((2, 1))
        V = np.ones((4, 1))
        W = np.ones((2, 1))
        return U, V, W
    
    # Fallback: schoolbook
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    size_u = n * m  # 12
    size_v = m * p  # 24
    size_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # First, get a structured decomposition
    U_struct, V_struct, W_struct = _block_decompose(n, m, p)
    r_struct = U_struct.shape[1]
    
    # Verify structured decomposition
    if not _verify(U_struct, V_struct, W_struct, n, m, p, atol=1e-10):
        # Fall back to schoolbook
        r_struct = n * m * p
        U_struct = np.zeros((size_u, r_struct))
        V_struct = np.zeros((size_v, r_struct))
        W_struct = np.zeros((size_w, r_struct))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_struct[i * m + j, col] = 1.0
                    V_struct[j * p + k, col] = 1.0
                    W_struct[k * n + i, col] = 1.0
                    col += 1
    
    # Try to improve with numerical optimization starting from structured solution
    best_result = (r_struct, U_struct, V_struct, W_struct)
    
    # Try polishing the structured solution to potentially find lower rank
    # First, try to reduce rank by zeroing out some columns
    for target_r in range(max(20, r_struct - 10), r_struct + 1):
        if time.time() > deadline:
            break
        
        # Start from structured solution, keep first target_r columns
        if target_r >= r_struct:
            U_init = U_struct.copy()
            V_init = V_struct.copy()
            W_init = W_struct.copy()
        else:
            # Use a random subset or try to merge
            np.random.seed(42 + target_r)
            U_init = U_struct[:, :target_r] + np.random.randn(size_u, target_r) * 0.1
            V_init = V_struct[:, :target_r] + np.random.randn(size_v, target_r) * 0.1
            W_init = W_struct[:, :target_r] + np.random.randn(size_w, target_r) * 0.1
        
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[0:size_u*target_r].reshape(size_u, target_r)
            V = params[size_u*target_r:size_u*target_r + size_v*target_r].reshape(size_v, target_r)
            W = params[size_u*target_r + size_v*target_r:].reshape(size_w, target_r)
            
            D = np.zeros((n, m, p))
            for c in range(target_r):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
            
            return (D - 1.0).flatten()
        
        try:
            result = least_squares(residuals, params0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12)
            
            if result.cost < 1e-8:
                U_sol = result.x[0:size_u*target_r].reshape(size_u, target_r)
                V_sol = result.x[size_u*target_r:size_u*target_r + size_v*target_r].reshape(size_v, target_r)
                W_sol = result.x[size_u*target_r + size_v*target_r:].reshape(size_w, target_r)
                
                # Snap to nearest half-integer
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if _verify(U_snap, V_snap, W_snap, n, m, p, atol=1e-8):
                    best_result = (target_r, U_snap, V_snap, W_snap)
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        except Exception:
            pass
    
    # Return best found
    r_best, U_best, V_best, W_best = best_result
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
