"""Matrix multiplication 3,4,8 tensor decomposition with Kronecker structure."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 145
    
    rank_trivial = n * m * p  # 96
    triples = [(i, k, j) for i in range(n) for k in range(m) for j in range(p)]
    T = len(triples)
    
    idx_U = np.array([i*m+k for i,k,j in triples])
    idx_V = np.array([k*p+j for i,k,j in triples])
    idx_W = np.array([j*n+i for i,k,j in triples])
    
    def _verify_fast(U, V, W, tol=1e-9):
        r = U.shape[1]
        for t in range(T):
            val = np.dot(U[idx_U[t], :], V[idx_V[t], :] * W[idx_W[t], :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def _is_half_integer(arr, tol=1e-6):
        return bool(np.all(np.abs(arr * 2 - np.round(arr * 2)) < tol))
    
    def _to_output(U, V, W):
        U_out = [[float(U[i,r]) for r in range(U.shape[1])] for i in range(U.shape[0])]
        V_out = [[float(V[i,r]) for r in range(V.shape[1])] for i in range(V.shape[0])]
        W_out = [[float(W[i,r]) for r in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    best_result = None
    best_rank = rank_trivial
    
    def _trivial():
        U_out = [[0]*rank_trivial for _ in range(n*m)]
        V_out = [[0]*rank_trivial for _ in range(m*p)]
        W_out = [[0]*rank_trivial for _ in range(p*n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i*(m*p) + k*p + j
                    U_out[i*m+k][r] = 1
                    V_out[k*p+j][r] = 1
                    W_out[j*n+i][r] = 1
        return {"U": U_out, "V": V_out, "W": W_out}
    
    def _kronecker_init(rank):
        """Initialize using Kronecker product structure.
        3x4x8 = (3x1x4) ⊗ (1x4x2) roughly, or other decompositions.
        We use: C[i,j] = sum_k A[i,k] B[k,j]
        Split k into k1 (size 2) and k2 (size 2): k = 2*k1 + k2
        Split j into j1 (size 4) and j2 (size 2): j = 2*j1 + j2
        Then C[i, 2*j1+j2] = sum_{k1,k2} A[i, 2*k1+k2] B[2*k1+k2, 2*j1+j2]
        
        This doesn't directly give Kronecker structure. Let me try another approach.
        
        Actually, let's use: 3x4x8 = (3x4x1) ⊗ (1x1x8) = rank 12 * 8 = 96, not helpful.
        Or: 3x4x8 = (3x1x8) ⊗ (1x4x1) = rank 8 * 4 = 32.
        
        For (3x1x8): C[i,j] = A[i] * B[j], rank 8 (each i,j pair needs its own term)
        For (1x4x1): C = A * B, rank 4
        
        So Kronecker gives rank 32.
        """
        # 3x1x8 decomposition: rank 8
        # U1: (3*1, 8) = (3, 8), V1: (1*8, 8) = (8, 8), W1: (8*3, 8) = (24, 8)
        # For 3x1x8: C[i,j] = A[i] * B[j]
        # U1[i, r] = 1 if r == i else 0 (for i in 0..2)
        # V1[j, r] = 1 if r == j else 0 (for j in 0..7)  
        # W1[j*3+i, r] = 1 if r == i else 0 ... wait this doesn't work simply.
        
        # Let me just do the standard rank-1 decomposition for 3x1x8:
        # C[i,j] = A[i] * B[j] = sum_r U1[i,r] V1[j,r] W1[j*3+i, r]
        # Use: U1[i,r] = 1 if r==i else 0, V1[j,r] = 1 if r==j else 0, W1[j*3+i,r] = 1 if r==i AND r==j... no.
        
        # Actually for outer product: C[i,j] = A[i]*B[j]
        # This is a rank-1 tensor in the (i,j) space with values A[i]*B[j].
        # As a bilinear form: sum_r U[i,r] V[j,r] W[j*3+i,r] where we need U[i,r]*V[j,r]*W[j*3+i,r] = delta_{i,j} for the tensor entries.
        # The tensor entries for 3x1x8: T[i,0,j] = 1 for all i,j. So it's just the all-ones tensor.
        # Rank of all-ones 3x8 matrix as a tensor: it's rank 1! Because T[i,k,j] = 1 = U[i,0]*V[k,0]*W[j,0] with all ones.
        # Wait, for 3x1x8: T[i,k,j] where k only has value 0. T[i,0,j] = 1.
        # This is indeed rank 1: U[i,0]=1, V[0,0]=1, W[j,0]=1 for all i,j.
        
        # So 3x1x8 has rank 1! And 1x4x8 has rank 4 (trivial for 1x4x8: C[j] = sum_k A[k]*B[k,j], which is rank 4).
        # Kronecker: rank(3x4x8) <= rank(3x1x8) * rank(1x4x8) = 1 * 4 = 4? No, that's not right.
        
        # The Kronecker product of tensors: if T1 has rank r1 and T2 has rank r2, then T1⊗T2 has rank <= r1*r2.
        # 3x4x8 = (3x1x1) ⊗ (1x4x8)? No, dimensions don't match.
        # 3x4x8: first dim 3, second 4, third 8.
        # We can write 3 = 3*1, 4 = 1*4, 8 = 8*1. So 3x4x8 = (3x1x8) ⊗ (1x4x1).
        # rank(3x1x8) = 1 (as shown above), rank(1x4x1) = 4.
        # So rank(3x4x8) <= 1*4 = 4? That can't be right.
        
        # Let me reconsider. The tensor for 3x4x8 multiplication:
        # T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
        # The tensor entries are T[i,k,j] = 1 for all i,k,j (it's the "multiplication tensor").
        # Wait no. The multiplication tensor has T[i,k,j] = 1 for all valid (i,k,j), and the decomposition is:
        # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # where T[i,k,j] should equal 1 for all i,k,j.
        
        # So we need: for all i,k,j: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
        # And for all other index combinations (which don't exist in the tensor), the product should be 0.
        # But since we only have indices that correspond to valid (i,k,j) triples, we just need the above.
        
        # OK so the problem is: find U (12 x r), V (32 x r), W (24 x r) such that
        # for all (i,k,j) in range(3)xrange(4)xrange(8):
        #   sum_r U[i*4+k, r] * V[k*8+j, r] * W[j*3+i, r] = 1
        
        # The trivial solution has r=96 with one-hot vectors.
        # The Kronecker structure: if we can write this as a tensor product...
        
        # Let me just try random initialization with the constraint structure and optimize.
        
        if rank >= 32:
            # Try Kronecker-like initialization
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Use the fact that we can decompose into blocks
            # For each i (0..2), k (0..3), j (0..7):
            # We want sum_r U[i*4+k, r] * V[k*8+j, r] * W[j*3+i, r] = 1
            
            # Simple initialization: spread out the contributions
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        # Assign a small contribution from each rank component
                        for r in range(rank):
                            U[i*m+k, r] += 1.0 / rank
                            V[k*p+j, r] += 1.0 / np.sqrt(rank)  
                            W[j*n+i, r] += 1.0 / np.sqrt(rank)
            return U, V, W
        else:
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            return U, V, W
    
    def _optimize(U, V, W, max_iters=200, lr=0.01):
        """Optimize with adaptive learning rate."""
        n, m, p = DIMENSIONS
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        # Precompute expanded indices
        U_exp = U[idx_U, :]
        V_exp = V[idx_V, :]
        W_exp = W[idx_W, :]
        
        best_loss = np.inf
        patience = 0
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_exp = U[idx_U, :]
            V_exp = V[idx_V, :]
            W_exp = W[idx_W, :]
            
            errors = np.sum(U_exp * V_exp * W_exp, axis=1) - 1
            loss = np.sum(errors**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 30:
                break
            
            if loss < 1e-10:
                break
            
            # Adaptive learning rate
            effective_lr = lr * max(0.1, 1.0 / (1.0 + loss))
            
            grad_U_exp = 2 * errors[:, None] * (V_exp * W_exp)
            grad_V_exp = 2 * errors[:, None] * (U_exp * W_exp)
            grad_W_exp = 2 * errors[:, None] * (U_exp * V_exp)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            np.add.at(grad_U, idx_U, grad_U_exp)
            np.add.at(grad_V, idx_V, grad_V_exp)
            np.add.at(grad_W, idx_W, grad_W_exp)
            
            # Gradient clipping
            max_grad = np.max(np.abs(grad_U)) + np.max(np.abs(grad_V)) + np.max(np.abs(grad_W))
            if max_grad > 10:
                scale = 10.0 / max_grad
                grad_U *= scale
                grad_V *= scale
                grad_W *= scale
            
            U -= effective_lr * grad_U
            V -= effective_lr * grad_V
            W -= effective_lr * grad_W
        
        if best_loss < loss:
            return best_U, best_V, best_W, best_loss
        return U, V, W, loss
    
    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    lower_bound = max(n*m, m*p, p*n)
    
    # First, try the Kronecker-based rank 32
    for rank in [32, 33, 34, 35, 36, 38, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, rank_trivial]:
        if time.time() > deadline:
            break
        if rank < lower_bound:
            continue
        
        found = False
        attempts = 10 if rank <= 40 else 5
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            U, V, W = _kronecker_init(rank)
            U, V, W, loss = _optimize(U, V, W, max_iters=150, lr=0.02)
            
            if loss < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U*2)/2
                V_r = np.round(V*2)/2
                W_r = np.round(W*2)/2
                
                if _verify_fast(U_r, V_r, W_r, 1e-8):
                    best_result = _to_output(U_r, V_r, W_r)
                    best_rank = rank
                    found = True
                    break
                
                # If rounding doesn't work, try more precise optimization
                U, V, W, loss = _optimize(U_r, V_r, W_r, max_iters=100, lr=0.005)
                U_r2 = np.round(U*2)/2
                V_r2 = np.round(V*2)/2
                W_r2 = np.round(W*2)/2
                
                if _verify_fast(U_r2, V_r2, W_r2, 1e-8):
                    best_result = _to_output(U_r2, V_r2, W_r2)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result:
        return best_result
    
    # Fallback: try a few more ranks with random init
    for rank in range(lower_bound, min(lower_bound + 20, rank_trivial + 1)):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            U, V, W, loss = _optimize(U, V, W, max_iters=100, lr=0.01)
            
            if loss < 1e-6:
                U_r = np.round(U*2)/2
                V_r = np.round(V*2)/2
                W_r = np.round(W*2)/2
                if _verify_fast(U_r, V_r, W_r, 1e-8):
                    best_result = _to_output(U_r, V_r, W_r)
                    best_rank = rank
                    break
        if best_result:
            break
    
    if best_result:
        return best_result
    
    return _trivial()
# EVOLVE-BLOCK-END
