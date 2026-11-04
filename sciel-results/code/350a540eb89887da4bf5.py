"""Improved program for 4x4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, leaving 20s margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    def verify(U, V, W, tol=1e-8):
        """Verify that U, V, W exactly decompose the tensor."""
        # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # Check all non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check some zero entries to ensure no spurious contributions
        # Sample a few zero entries
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this is a zero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if not (i == i2 and j == j2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Try structured decomposition approaches
    # Approach 1: Use Kronecker product structure
    # The 4x4x8 tensor can be seen as two 4x4x4 tensors (for k=0..3 and k=4..7)
    # If we have a rank-r decomposition for 4x4x4, we can get rank 2r for 4x4x8
    # The known minimal rank for 4x4x4 is 20, so this gives rank 40
    
    # Let's try to find a rank-20 decomposition for 4x4x4 first
    # Then combine for 4x4x8
    
    # For 4x4x4: T4[a,b,c] where a=i*4+j, b=j*4+k, c=k*4+i, all in 0..15
    # We need U4 (16 x r), V4 (16 x r), W4 (16 x r) such that
    # sum_d U4[a,d]*V4[b,d]*W4[c,d] = T4[a,b,c]
    
    # Try numerical search for 4x4x4 rank-20 decomposition
    def find_4x4x4_decomp(target_rank, max_time=30):
        """Find a rank-target_rank decomposition for 4x4x4."""
        n4 = m4 = p4 = 4
        T4 = np.zeros((n4 * m4, m4 * p4, p4 * n4), dtype=np.float64)
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    a = i * m4 + j
                    b = j * p4 + k
                    c = k * n4 + i
                    T4[a, b, c] = 1.0
        
        best_U4 = None
        best_V4 = None
        best_W4 = None
        
        for trial in range(100):
            if time.time() > start_time + max_time:
                break
            
            U4 = np.random.randn(n4 * m4, target_rank)
            V4 = np.random.randn(m4 * p4, target_rank)
            W4 = np.random.randn(p4 * n4, target_rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > start_time + max_time:
                    break
                
                # Update U4
                for a in range(n4 * m4):
                    num = np.sum(T4[a, :, :] * (V4 * W4.T), axis=(0, 1))
                    den = np.sum((V4 * V4) * (W4 * W4).T, axis=(0, 1))
                    U4[a, :] = num / (den + 1e-10)
                
                # Update V4
                for b in range(m4 * p4):
                    num = np.sum(T4[:, b, :] * (U4 * W4.T), axis=(0, 1))
                    den = np.sum((U4 * U4) * (W4 * W4).T, axis=(0, 1))
                    V4[b, :] = num / (den + 1e-10)
                
                # Update W4
                for c in range(p4 * n4):
                    num = np.sum(T4[:, :, c] * (U4 * V4.T), axis=(0, 1))
                    den = np.sum((U4 * U4) * (V4 * V4).T, axis=(0, 1))
                    W4[c, :] = num / (den + 1e-10)
            
            # Check error
            T4_rec = np.einsum('ad,bd,cd->abc', U4, V4, W4)
            error = np.sum((T4 - T4_rec)**2)
            
            if error < 1e-6:
                best_U4 = U4
                best_V4 = V4
                best_W4 = W4
                break
        
        return best_U4, best_V4, best_W4
    
    # Try to find rank-20 decomposition for 4x4x4
    U4, V4, W4 = find_4x4x4_decomp(20, max_time=25)
    
    if U4 is not None:
        # Combine two 4x4x4 decompositions for 4x4x8
        # For k=0..3: use U4, V4, W4
        # For k=4..7: use U4, V4, W4 with shifted indices
        # 
        # U has shape (16, 40): first 20 cols for k=0..3, next 20 for k=4..7
        # V has shape (32, 40): 
        #   For k=0..3: V[j*8+k, col] = V4[j*4+k, col] for col=0..19
        #   For k=4..7: V[j*8+k, col+20] = V4[j*4+(k-4), col] for col=0..19
        # W has shape (32, 40):
        #   For k=0..3: W[k*4+i, col] = W4[k*4+i, col] for col=0..19
        #   For k=4..7: W[k*4+i, col+20] = W4[(k-4)*4+i, col] for col=0..19
        
        r4 = U4.shape[1]
        total_rank = 2 * r4
        
        U = np.zeros((n * m, total_rank), dtype=np.float64)
        V = np.zeros((m * p, total_rank), dtype=np.float64)
        W = np.zeros((p * n, total_rank), dtype=np.float64)
        
        # First half: k=0..3
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a4 = i * 4 + j
                U[a, 0:r4] = U4[a4, :]
        
        for j in range(m):
            for k in range(4):
                b = j * p + k
                b4 = j * 4 + k
                V[b, 0:r4] = V4[b4, :]
        
        for k in range(4):
            for i in range(n):
                c = k * n + i
                c4 = k * 4 + i
                W[c, 0:r4] = W4[c4, :]
        
        # Second half: k=4..7
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a4 = i * 4 + j
                U[a, r4:2*r4] = U4[a4, :]
        
        for j in range(m):
            for k in range(4):
                b = j * p + (4 + k)
                b4 = j * 4 + k
                V[b, r4:2*r4] = V4[b4, :]
        
        for k in range(4):
            for i in range(n):
                c = (4 + k) * n + i
                c4 = k * 4 + i
                W[c, r4:2*r4] = W4[c4, :]
        
        # Verify
        if verify(U, V, W):
            best_U = U
            best_V = V
            best_W = W
            best_rank = total_rank
    
    # If we didn't find a good decomposition, try direct numerical search for 4x4x8
    if best_rank >= 100:
        # Try direct ALS on the full 4x4x8 tensor
        for target_rank in [40, 45, 50]:
            if time.time() > deadline:
                break
            
            for trial in range(50):
                if time.time() > deadline:
                    break
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # ALS iterations
                for it in range(80):
                    if time.time() > deadline:
                        break
                    
                    # Update U
                    for a in range(n * m):
                        num = np.sum(T[a, :, :] * (V * W.T), axis=(0, 1))
                        den = np.sum((V * V) * (W * W).T, axis=(0, 1))
                        U[a, :] = num / (den + 1e-10)
                    
                    # Update V
                    for b in range(m * p):
                        num = np.sum(T[:, b, :] * (U * W.T), axis=(0, 1))
                        den = np.sum((U * U) * (W * W).T, axis=(0, 1))
                        V[b, :] = num / (den + 1e-10)
                    
                    # Update W
                    for c in range(p * n):
                        num = np.sum(T[:, :, c] * (U * V.T), axis=(0, 1))
                        den = np.sum((U * U) * (V * V).T, axis=(0, 1))
                        W[c, :] = num / (den + 1e-10)
                
                # Check error
                T_rec = np.einsum('ad,bd,cd->abc', U, V, W)
                error = np.sum((T - T_rec)**2)
                
                if error < 1e-6:
                    if verify(U, V, W):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = target_rank
                        break
            
            if best_rank <= target_rank:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
