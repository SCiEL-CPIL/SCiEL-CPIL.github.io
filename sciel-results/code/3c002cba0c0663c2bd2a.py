"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # The matrix multiplication tensor T[i,j,k] = e_{i,j} * e_{k,i} for the W[k*n+i] convention
    # We need U (nm x r), V (mp x r), W (pn x r) such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other combinations it equals 0 (which is automatically satisfied if we set it up correctly)
    
    # Schoolbook rank is n*m*p = 42. Let's try to find a lower rank.
    # The known lower bounds for 2x3 x 3x7 are not trivial.
    # Let's use a randomized approach: start with a random rank r decomposition and try to optimize.
    # Or better, use the fact that the tensor is defined and try to find a rank-r decomposition via optimization.
    
    # Strategy: Try ranks from a low value up to 42. For each rank r, initialize U,V,W randomly,
    # and use a few iterations of gradient-free or gradient-based optimization to try to fit the tensor.
    # However, exact decomposition requires integer/half-integer entries, so we need to round at the end.
    
    # A more practical approach: Since the problem size is small, we can try to use the known structure.
    # For 2x3 x 3x7, the rank is at most 42. It might be possible to reduce it.
    # But given the time constraint and the difficulty of exact tensor rank minimization,
    # let's first try to see if we can find a rank less than 42 with a simple heuristic.
    
    # Actually, the most reliable "improvement" over schoolbook (42) would be to find any rank < 42.
    # If we can't find one quickly, we return the schoolbook decomposition.
    
    # Let's try a few random initializations with rank 40, 38, 36... and optimize.
    # We'll use a simple gradient descent on the squared error of the tensor reconstruction.
    
    def build_tensor(U, V, W):
        """Reconstruct the tensor from U, V, W."""
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            # U[:, c] is a vector of length nm=6, representing u_{i,j}
            # V[:, c] is a vector of length mp=21, representing v_{j,k}
            # W[:, c] is a vector of length pn=14, representing w_{k,i}
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # The contribution to T[i,j,k] is u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return T
    
    def tensor_error(U, V, W):
        """Compute the squared error of the tensor reconstruction."""
        T = build_tensor(U, V, W)
        # The target tensor has T[i,j,k] = 1 if the index combination corresponds to the multiplication, 0 otherwise.
        # Actually, the matrix multiplication tensor is defined such that the product of two matrices A (n x m) and B (m x p)
        # gives C (n x p) with C[i,k] = sum_j A[i,j] * B[j,k].
        # The tensor T[i,j,k] = 1 if we're computing the (i,k) entry using the (i,j) and (j,k) entries, which is always 1 for valid i,j,k.
        # Wait, the tensor is a 3-way tensor of size n x m x p where T[i,j,k] = 1 for all i,j,k? No, that's not right.
        # The matrix multiplication tensor is a 3-way tensor where the mode-1 index is (i), mode-2 is (j), mode-3 is (k).
        # The value T[i,j,k] = 1 if the term A[i,j] * B[j,k] appears in C[i,k], which it does for all i,j,k.
        # So the target tensor is simply all ones? No, that's not the tensor that we're decomposing.
        # The tensor we're decomposing is the structure tensor of matrix multiplication.
        # It's a tensor T in U ⊗ V ⊗ W where U is the space of n x m matrices, V is the space of m x p matrices, W is the space of n x p matrices.
        # The tensor is T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with the W index being k*n+i).
        # So the target tensor is a 3-way tensor of size (n*m) x (m*p) x (p*n) with 1's at specific positions.
        
        # Let's re-examine the convention. The problem states W[k*n+i, r].
        # The decomposition is: for each r, U[:, r] is a vector in R^{n*m}, V[:, r] is a vector in R^{m*p}, W[:, r] is a vector in R^{p*n}.
        # The condition is that the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        # And this should be 0 for any other combination of indices, but since the indices are fixed to i,j,k, the condition is just that the sum equals 1.
        
        # So the target is a 3-way tensor T of size (n*m) x (m*p) x (p*n) where T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
        # But note that for a given i,j,k, the indices are fixed: a = i*m+j, b = j*p+k, c = k*n+i.
        # So the target tensor has 1's at these specific positions and 0's elsewhere.
        
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        # Compute the reconstructed tensor
        # The reconstructed tensor R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We can compute this efficiently:
        # R = U @ (V * W) where V*W is element-wise? No, that's not right.
        # R[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This is like a tensor product. We can compute it as:
        # For each r, we have a rank-1 tensor U[:, r] ⊗ V[:, r] ⊗ W[:, r]
        # So R = sum_r outer(outer(U[:, r], V[:, r]), W[:, r])
        
        R = np.zeros((nm, mp, pn))
        for r in range(U.shape[1]):
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.sum((R - target) ** 2)
    
    def verify(U, V, W):
        """Check if the decomposition is exact (within tolerance) and has integer/half-integer entries."""
        r = U.shape[1]
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > 1e-6:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < 1e-6):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    def compute_gradient_U(U, V, W, target):
        """Compute gradient of tensor_error w.r.t. U."""
        grad_U = np.zeros_like(U)
        r = U.shape[1]
        for c in range(r):
            # R[:, :, c] = U[:, c] * (V * W) where V*W is outer product per column
            # Actually, R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # d/dU[a,c'] sum_{a,b,c'} (R[a,b,c'] - target[a,b,c'])^2 
            # For fixed c', the gradient w.r.t. U[a,c'] is:
            # 2 * sum_{b,c} (R[a,b,c] - target[a,b,c]) * V[b,c'] * W[c,c']
            # But we need to be careful: R depends on all U[:,c]
            
            # Let's compute R for all c, then compute gradient
            pass
        
        # Better approach: compute R = sum_c outer(outer(U[:,c], V[:,c]), W[:,c])
        # Then grad_U[a, c'] = 2 * sum_{b,c} (R[a,b,c] - target[a,b,c]) * V[b,c'] * W[c,c']
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        error_tensor = R - target
        
        for c in range(r):
            # grad_U[:, c] = 2 * einsum('abc,bc->ac', error_tensor, V[:, c] * W[:, c])
            # But V[:, c] * W[:, c] doesn't make sense dimensionally
            # grad_U[a, c] = 2 * sum_{b,c2} error_tensor[a,b,c2] * V[b,c] * W[c2,c]
            # Wait, W index is c2 for the third mode, but we're differentiating w.r.t. U[:, c]
            # R[a,b,c2] = sum_r U[a,r] * V[b,r] * W[c2,r]
            # dR[a,b,c2]/dU[a',c] = delta_{a,a'} * V[b,c] * W[c2,c]
            # So grad_U[a',c] = 2 * sum_{b,c2} error_tensor[a',b,c2] * V[b,c] * W[c2,c]
            
            # Compute this efficiently:
            # temp[b, c2] = error_tensor[a', b, c2] * W[c2, c]
            # Then grad_U[a', c] = 2 * sum_b temp[b, :] * V[b, c]
            # = 2 * (error_tensor * W[:, :, c]) @ V[:, c]  -- but need to be careful with dimensions
            
            # error_tensor has shape (nm, mp, pn)
            # W[:, c] has shape (pn,)
            # We want: for each a', sum_{b,c2} error_tensor[a',b,c2] * W[c2,c] * V[b,c]
            # = sum_{b,c2} error_tensor[a',b,c2] * (W[:, c] * V[:, c] broadcast appropriately)
            
            # Let's do it directly:
            # grad_U[:, c] = 2 * np.einsum('abc,bc->ac', error_tensor, np.einsum('b,c->bc', V[:, c], np.ones(pn)) * np.einsum('c2,c->c2c', W[:, c], np.ones(mp)))
            # This is getting complicated. Let's just do it with loops for small sizes.
            
            for a in range(nm):
                grad = 0.0
                for b in range(mp):
                    for c2 in range(pn):
                        grad += error_tensor[a, b, c2] * V[b, c] * W[c2, c]
                grad_U[a, c] = 2.0 * grad
        
        return grad_U
    
    def compute_gradient_V(U, V, W, target):
        """Compute gradient of tensor_error w.r.t. V."""
        grad_V = np.zeros_like(V)
        r = U.shape[1]
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        error_tensor = R - target
        
        for c in range(r):
            for b in range(mp):
                grad = 0.0
                for a in range(nm):
                    for c2 in range(pn):
                        grad += error_tensor[a, b, c2] * U[a, c] * W[c2, c]
                grad_V[b, c] = 2.0 * grad
        
        return grad_V
    
    def compute_gradient_W(U, V, W, target):
        """Compute gradient of tensor_error w.r.t. W."""
        grad_W = np.zeros_like(W)
        r = U.shape[1]
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        error_tensor = R - target
        
        for c in range(r):
            for c2 in range(pn):
                grad = 0.0
                for a in range(nm):
                    for b in range(mp):
                        grad += error_tensor[a, b, c2] * U[a, c] * V[b, c]
                grad_W[c2, c] = 2.0 * grad
        
        return grad_W
    
    def optimize(U, V, W, max_iters=200, lr_init=0.01):
        """Optimize U, V, W to minimize tensor error using gradient descent."""
        lr = lr_init
        prev_error = tensor_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            grad_U = compute_gradient_U(U, V, W, target)
            grad_V = compute_gradient_V(U, V, W, target)
            grad_W = compute_gradient_W(U, V, W, target)
            
            # Update
            U_new = U - lr * grad_U
            V_new = V - lr * grad_V
            W_new = W - lr * grad_W
            
            new_error = tensor_error(U_new, V_new, W_new)
            
            if new_error < prev_error:
                U, V, W = U_new, V_new, W_new
                prev_error = new_error
                # Decay learning rate slowly
                lr *= 0.999
            else:
                # Reduce learning rate and try again
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        return U, V, W, prev_error
    
    best_result = None
    best_rank = 42
    
    # Try ranks from 30 up to 41
    for rank in range(30, 42):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(hash((rank, attempt)) % 2**32)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Optimize
            U_opt, V_opt, W_opt, final_error = optimize(U, V, W, max_iters=100, lr_init=0.05)
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U_opt)
            V_rounded = round_to_half_integer(V_opt)
            W_rounded = round_to_half_integer(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_result = (U_rounded, V_rounded, W_rounded)
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    # Convert to lists as required
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
