"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # We want to find rank-minimal decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # with W[k*n+i, r] following the trace convention
    
    # Start with schoolbook rank = n*m*p = 96
    rank_schoolbook = n * m * p
    
    # Try to find lower rank decompositions
    # Known results:
    # For 3x4*4x8, the minimal rank is at most 96 (schoolbook)
    # Strassen-type algorithms might help but are complex
    # Let's try randomized search for lower rank
    
    # First, let's try a systematic approach:
    # The tensor is T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k
    # This is the matrix multiplication tensor
    
    # For small dimensions, we can try to find exact decompositions
    # using integer or half-integer entries
    
    # Try ranks from lower to higher
    # Known lower bounds: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # So rank >= 32
    
    # Let's try to find a decomposition with rank < 96
    # Use randomized local search
    
    def random_init(rank, scale=1.0):
        """Initialize random factor matrices."""
        U = np.random.randn(n*m, rank) * scale
        V = np.random.randn(m*p, rank) * scale
        W = np.random.randn(p*n, rank) * scale
        return U, V, W
    
    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        tensor = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n*m):
                if u[i] != 0:
                    for j in range(m*p):
                        if v[j] != 0:
                            for k in range(p*n):
                                if w[k] != 0:
                                    tensor[i, j, k] += u[i] * v[j] * w[k]
        return tensor
    
    def tensor_error(U, V, W):
        """Compute the error between the reconstructed tensor and the target."""
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute reconstructed tensor efficiently
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Outer product
            recon += np.einsum('a,b,c->abc', u, v, w)
        
        return np.linalg.norm(target - recon)
    
    def snap_to_half_integers(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            recon += np.einsum('a,b,c->abc', u, v, w)
        
        return np.allclose(target, recon, atol=1e-10)
    
    # Try to find a lower rank decomposition
    best_rank = rank_schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Initialize with schoolbook
    U_init = np.zeros((n*m, rank_schoolbook))
    V_init = np.zeros((m*p, rank_schoolbook))
    W_init = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, col] = 1
                V_init[j*p+k, col] = 1
                W_init[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = rank_schoolbook
    
    # Try to reduce rank using local search
    # Start from a random initialization with lower rank
    random.seed(42)
    
    for target_rank in [64, 56, 48, 40, 36, 32]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            U, V, W = random_init(target_rank, scale=0.5)
            
            # Local search: iteratively update one factor at a time
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n*m):
                    # Optimize U[i, :] given V and W
                    # We want sum_r U[i,r] * V[:,r] * W[:,r] to match target[i, :, :]
                    # This is a linear system: U[i,:] * (V * W outer) = target[i,:,:]
                    # Actually, for fixed i, we need:
                    # sum_r U[i,r] * V[j,r] * W[k,r] = target[i,j,k] for all j,k
                    # This can be written as: U[i,:] @ M = target[i,:,:].reshape(-1)
                    # where M[j*p+k, r] = V[j,r] * W[k,r]
                    
                    M = np.zeros((m*p*p*n, target_rank))
                    for j in range(m*p):
                        for k in range(p*n):
                            M[j*p*n+k, :] = V[j, :] * W[k, :]
                    
                    target_i = target_flat[i] if 'target_flat' in dir() else None
                    # Compute target for this i
                    target_i = np.zeros(m*p*p*n)
                    for j in range(m*p):
                        for k in range(p*n):
                            # Check if this corresponds to a valid (i,j,k) in the tensor
                            # i = i, j = j, k = k where i*m+j is the first index
                            # j*p+k is the second index, k*n+i is the third
                            ii = i
                            jj = j
                            kk = k
                            # The tensor entry is 1 if jj//p == ii%m and kk//n == ii//m
                            # Wait, let me re-derive:
                            # First index: i*m + j where i in [0,n), j in [0,m)
                            # Second index: j*p + k where j in [0,m), k in [0,p)
                            # Third index: k*n + i where k in [0,p), i in [0,n)
                            # So for fixed first index i*m+j0, we have i=i, j=j0
                            # Second index j1*p+k1 must have j1=j0
                            # Third index k2*n+i2 must have i2=i
                            # So j1 = j0, i2 = i, and k1 = k2
                            if jj // p == ii % m and kk // n == ii // m:
                                target_i[jj*p*n+kk] = 1.0
                    
                    # Solve least squares
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target_i, rcond=None)
                        U[i, :] = sol
                    except:
                        pass
                
                # Update V
                for j in range(m*p):
                    M = np.zeros((n*m*p*n, target_rank))
                    for i in range(n*m):
                        for k in range(p*n):
                            M[i*p*n+k, :] = U[i, :] * W[k, :]
                    
                    target_j = np.zeros(n*m*p*n)
                    for i in range(n*m):
                        for k in range(p*n):
                            ii = i
                            jj = j
                            kk = k
                            if jj // p == ii % m and kk // n == ii // m:
                                target_j[i*p*n+kk] = 1.0
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target_j, rcond=None)
                        V[j, :] = sol
                    except:
                        pass
                
                # Update W
                for k in range(p*n):
                    M = np.zeros((n*m*m*p, target_rank))
                    for i in range(n*m):
                        for j in range(m*p):
                            M[i*m*p+j, :] = U[i, :] * V[j, :]
                    
                    target_k = np.zeros(n*m*m*p)
                    for i in range(n*m):
                        for j in range(m*p):
                            ii = i
                            jj = j
                            kk = k
                            if jj // p == ii % m and kk // n == ii // m:
                                target_k[i*m*p+jj] = 1.0
                    
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target_k, rcond=None)
                        W[k, :] = sol
                    except:
                        pass
            
            # Check if we found a valid decomposition
            if verify_decomposition(U, V, W):
                # Snap to half-integers
                U_snap = snap_to_half_integers(U)
                V_snap = snap_to_half_integers(V)
                W_snap = snap_to_half_integers(W)
                if verify_decomposition(U_snap, V_snap, W_snap):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                    break
                else:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                    break
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
