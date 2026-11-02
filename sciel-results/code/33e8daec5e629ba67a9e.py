import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.linalg import sqrtm

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Precompute target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Else 0
    # We want U[a, r] V[b, r] W[c, r] summed over r to equal T[a, b, c]
    
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_set = set(target_indices)
    
    # The number of non-zero entries in the tensor is n*m*p = 96
    # The total number of entries in the tensor space is nU*nV*nW = 12*32*24 = 9216
    # This is too large for a direct SDP with all constraints.
    # However, we can use the "Gram Matrix" approach.
    # Let G be a symmetric positive semidefinite matrix of size R x R.
    # We don't know R. But we can fix a candidate R and solve for G.
    # Actually, the standard SDP relaxation for tensor rank is:
    # Minimize trace(G)
    # Subject to:
    # G >= 0
    # For all (a,b,c) in support(T): sum_r G[r,r] ... wait, this is not quite right.
    
    # A better formulation for "Bilinear Rank" or "Tensor Rank" via SDP:
    # We are looking for U, V, W such that T = sum_r u_r v_r^T w_r^T (outer product).
    # This is equivalent to finding a set of vectors.
    # There is a known SDP relaxation:
    # Minimize trace(X)
    # Subject to X >= 0
    # And linear constraints derived from the tensor.
    
    # Let's try a different angle:
    # If we fix the rank R, we can try to find U, V, W.
    # The equations are:
    # For all a, b, c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # This is a system of quadratic equations.
    
    # Alternative: Use the fact that for a fixed R, we can flatten the problem.
    # Let's try to use scipy.optimize.minimize with a clever objective.
    # Objective: Sum of squared residuals.
    # Variables: U, V, W flattened.
    # This is the same as the current program's approach, just using a better optimizer.
    # The current program uses manual simulated annealing.
    # Let's use `scipy.optimize.minimize` with method 'L-BFGS-B' or 'Powell'.
    # But the landscape is non-convex.
    
    # Let's try a structural change: **Strassen-like block decomposition search**.
    # Instead of random entries, we search for a block structure.
    # For 3x4x8, maybe we can find a decomposition with rank < 96.
    # The theoretical lower bound is max(n*m, m*p, p*n) = 32.
    # Let's try to find a rank 32, 33, ... decomposition.
    
    # Strategy:
    # 1. Start with a candidate rank R.
    # 2. Initialize U, V, W with random values.
    # 3. Use a gradient-based optimizer (L-BFGS-B) to minimize the error.
    # 4. If error is close to 0, check validity.
    # 5. If valid, return. Else, try next R.
    
    # To make it more robust, we can try multiple initializations for each R.
    
    def get_flattened(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
    def get_matrices(x, R):
        size_U = nU * R
        size_V = nV * R
        size_W = nW * R
        U = x[:size_U].reshape(nU, R)
        V = x[size_U:size_U+size_V].reshape(nV, R)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(nW, R)
        return U, V, W
        
    def error_and_grad(x, R):
        U, V, W = get_matrices(x, R)
        
        # Compute tensor T_reconstructed
        # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # We only care about the target indices and a sample of non-target indices?
        # No, for exactness, we need all non-target indices to be 0.
        # But checking all 9216-96 = 9120 non-target indices is expensive.
        # However, if we minimize the sum of squares over ALL entries, it's 9216 terms.
        # 9216 * R operations per evaluation.
        # If R=32, that's ~300k ops. Very fast.
        
        # Let's compute the full tensor reconstruction.
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 12x32x24 array.
        # 12*32*24 = 9216 floats.
        
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Target tensor T
        # We can build T as a sparse representation or just a dense array.
        # Dense array is 9216 floats.
        T_target = np.zeros((nU, nV, nW))
        for a, b, c in target_indices:
            T_target[a, b, c] = 1.0
            
        diff = T_rec - T_target
        error = np.sum(diff**2)
        
        # Gradient
        # d/dU[a,r] (sum_{a',b',c'} (T_rec[a',b',c'] - T_target[a',b',c'])^2)
        # = 2 * sum_{b',c'} (T_rec[a,r,b',c'] - T_target[a,r,b',c']) * V[b',r] * W[c',r]
        # Using einsum for gradient:
        # grad_U[a,r] = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
        # grad_V[b,r] = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
        # grad_W[c,r] = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad

    # Try ranks from lower bound up to schoolbook
    schoolbook_rank = n * m * p
    min_rank_to_try = 32 # Lower bound
    max_rank_to_try = 60 # Let's not go too high initially
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Fallback: Schoolbook
    def build_schoolbook():
        U = np.zeros((nU, schoolbook_rank))
        V = np.zeros((nV, schoolbook_rank))
        W = np.zeros((nW, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W
        
    U_school, V_school, W_school = build_schoolbook()
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random starts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            x0 = np.random.randn(nU*rank + nV*rank + nW*rank) * 0.1
            
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    error_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'maxfun': 10000}
                )
            except Exception as e:
                continue
                
            if res.success or res.fun < 1e-10:
                U, V, W = get_matrices(res.x, rank)
                
                # Verify exactness
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                T_target = np.zeros((nU, nV, nW))
                for a, b, c in target_indices:
                    T_target[a, b, c] = 1.0
                
                if np.allclose(T_rec, T_target, atol=1e-9):
                    # Valid decomposition found
                    # Check if we can reduce rank further? 
                    # We are iterating up, so this is the first one found.
                    # But maybe a lower rank exists and we missed it?
                    # We started from 32. If we found one at 35, maybe 33 or 34 exists.
                    # But L-BFGS is local. 
                    # For now, let's accept the first valid one found in the sweep.
                    # To be safer, we could try all ranks up to this one with more attempts?
                    # But time is limited.
                    # Let's just return this one.
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
        # If not found, continue to next rank
        
    # If no lower rank found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# Import scipy at the top level to avoid import overhead inside the loop if possible, 
# but the prompt says standard library, numpy, scipy are available.
import scipy.optimize
