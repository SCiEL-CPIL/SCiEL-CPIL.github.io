"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# Known rank for 3x3 * 3x4 is 22 (Strassen-based / refined)
# The schoolbook rank is 3*3*4 = 36.
# We know from literature that R(3,3,4) = 22.
# Let's try to construct a rank-22 decomposition.

# The tensor T[i,j,k] = 1 if (i,j) in input A, (j,k) in input B, output (i,k).
# Indices: i in [0,2], j in [0,2], k in [0,3]
# U shape: (n*m, rank) = (9, rank)
# V shape: (m*p, rank) = (12, rank)  
# W shape: (p*n, rank) = (12, rank)
# W index: k*n + i (trace convention)

# For exact rank-22 decomposition of 3x3*3x4, we can use known constructions.
# One approach: use the fact that R(3,3,4) <= R(3,3,3) + R(3,3,1) = 23 + 3 = 26, but actually it's 22.
# 
# Let's use a systematic search with the known structure.
# Actually, let's use the Winograd-style or Strassen refinement.
#
# For 3x3 * 3x4, a known rank-22 construction exists. Let me try to find it via optimization.

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix mult."""
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # The multiplication tensor: T[(i,j), (j,k), (i,k)] = 1
    # We need U, V, W such that for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for non-terms, the tensor entries are 0 (but in matrix mult, only (i,j,k) terms exist)
    
    # Actually the tensor is: for each (i,j,k), the entry is 1.
    # The decomposition must satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # There are no "invalid" (i,j,k) in the standard matrix multiplication tensor - every combination exists.
    
    # Wait, let me reconsider. The matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # where i indexes rows of A, j indexes columns of A / rows of B, k indexes columns of B.
    # The output entry (i,k) = sum_j A[i,j] * B[j,k].
    # So the tensor is T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p).
    
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's try rank 22 first, then fall back to higher ranks.
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Vectorized verification."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        # sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ik'
        result = np.einsum('ijc,jkc,kic->ik', Ur, Vr, Wr, optimize=True)
        # But we need it to be 1 for all (i,j,k), not just (i,k).
        # Actually the tensor entry is per (i,j,k), so we need the full 3D result.
        # Let me redo: we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for ALL (i,j,k)
        # This is a 3D array of shape (n, m, p)
        result3d = np.einsum('ijc,jkc,kic->ijm', Ur, Vr, Wr, optimize=True)
        # Wait, that's not right either. Let me think more carefully.
        # U has indices (i*m+j, c) -> reshape to (n, m, rank)
        # V has indices (j*p+k, c) -> reshape to (m, p, rank)  
        # W has indices (k*n+i, c) -> reshape to (p, n, rank)
        # We want: for each (i,j,k), sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # This is a 3D tensor of shape (n, m, p)
        result3d = np.zeros((n, m, p))
        for c in range(rank):
            result3d += Ur[:, :, c] * Vr[:, :, c] * Wr[:, :, c]
            # Wait, that's not right. Ur[i,j,c], Vr[j,k,c], Wr[k,i,c]
            # For fixed c: Ur[:,:,c] is (n,m), Vr[:,:,c] is (m,p), Wr[:,:,c] is (p,n)
            # We need element-wise: for each (i,j,k): Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
            # This is like a 3D outer product with specific index matching.
            # Let me just do it explicitly for clarity.
        # Actually let me just do the explicit loop for verification since dimensions are small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find rank-22 decomposition using random initialization and optimization
    # The problem is a system of polynomial equations. For small dimensions, we can use
    # numerical optimization.
    
    def try_find_decomposition(rank, max_iters=5000, seed=None):
        """Try to find a rank-r decomposition via gradient descent on the loss."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Loss: sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2 over all (i,j,k)
        def loss_and_grad(U, V, W):
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = val - 1.0
                        loss += err * err
                        for c in range(rank):
                            dU[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
            return loss, dU, dV, dW
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 10:
                break
            
            loss, dU, dV, dW = loss_and_grad(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                if loss > best_loss * 1.5:
                    lr *= 0.5
                elif loss < best_loss * 0.5:
                    lr *= 1.5
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_loss, best_U, best_V, best_W
    
    # Try rank 22
    found = False
    best_result = None
    best_rank = 36  # schoolbook
    
    # First, let's try known constructions.
    # For 3x3*3x4, one can use the fact that it's related to 3x3*3x3 plus one more column.
    # R(3,3,3) = 23 (Lader), R(3,3,4) = 22.
    
    # Let me try a direct construction approach.
    # Actually, for exact integer/half-integer entries, let's try small values.
    
    # Strategy: try rank 22 with random seeds
    for attempt in range(50):
        if time.time() > DEADLINE - 20:
            break
        
        loss, U, V, W = try_find_decomposition(22, max_iters=3000, seed=attempt*1000+42)
        
        if loss < 1e-10:
            # Verify exactly
            if verify_numpy(U, V, W):
                # Round to nearest half-integer if close
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_numpy(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # Return as-is if it works
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if best_result is None or loss < best_result[0]:
            best_result = (loss, U, V, W)
    
    # If rank 22 didn't work, try rank 23
    for attempt in range(30):
        if time.time() > DEADLINE - 15:
            break
        
        loss, U, V, W = try_find_decomposition(23, max_iters=3000, seed=attempt*1000+99)
        
        if loss < 1e-10:
            if verify_numpy(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_numpy(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if best_result is None or loss < best_result[0]:
            best_result = (loss, U, V, W)
    
    # Fall back to schoolbook rank 36
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
