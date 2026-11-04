"""Rewritten program for 3x3x6 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 x 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (18, rank)  
    # W: (p*n, rank) = (18, rank)
    # Output: C[i,k] = sum_j A[i,j] B[j,k], stored as W[k*n+i, r]
    
    # The multiplication tensor T has entries T[(i,j), (j,k), (i,k)] = 1
    # where i in [0,n), j in [0,m), k in [0,p)
    # We want: T[(i,j), (j,k), (i,k)] = sum_r U[(i,j), r] * V[(j,k), r] * W[(i,k), r]
    
    # Flatten indices
    # A index: i*m + j, ranges over n*m = 9
    # B index: j*p + k, ranges over m*p = 18
    # C index: k*n + i, ranges over p*n = 18 (using trace convention W[k*n+i, r])
    
    # Build the tensor as a 3D array T[a, b, c] where:
    # a = i*m + j (A index)
    # b = j*p + k (B index)  
    # c = k*n + i (C index)
    # T[a, b, c] = 1 if the indices are consistent, 0 otherwise
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    # Known: schoolbook rank is n*m*p = 54
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But that's not a valid lower bound for tensor rank
    
    # Let's try to find the minimum rank
    # For exact decomposition, we need to solve:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of nonlinear equations. Let's try a greedy/iterative approach.
    
    # First, let's try the schoolbook rank as a baseline
    rank_schoolbook = n * m * p  # 54
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, T, tol=1e-10):
        """Check if U, V, W decompose T."""
        rank = U.shape[1]
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is a tensor contraction
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(result, T, atol=tol)
    
    if not verify(U_school, V_school, W_school, T):
        # Should always pass
        pass
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Use a simple iterative method: start with random initialization and optimize
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current tensor
            current = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T - current
            
            # Update U: for each a, U[a,:] should minimize sum_{b,c} residual[a,b,c] * V[b,:] * W[c,:]
            # This is: U[a,:] = argmin sum_{b,c} (T[a,b,c] - U[a,:]·(V[b,:]⊙W[c,:]))^2
            # Linear in U[a,:]: U[a,:] = (sum_{b,c} T[a,b,c] * (V[b,:]⊙W[c,:])) / (sum_{b,c} (V[b,:]⊙W[c,:])^2)
            
            # Precompute for efficiency
            # For each (b,c), compute V[b,:] * W[c,:] element-wise
            VW = np.einsum('br,cr->bc r', V, W)  # shape (m*p, p*n, r)
            
            # Update U
            for a in range(n*m):
                # sum over b,c of T[a,b,c] * VW[b,c,:]
                numerator = np.sum(T[a, :, :] * VW, axis=(0, 1))
                denominator = np.sum(VW[:, :, a] ** 2) if False else np.sum((VW * T[a, :, :][:, :, np.newaxis]) ** 2, axis=(0,1))
                # Actually, the correct update is:
                # U[a,:] = <T[a,:,:], VW> / <VW, VW> where the inner product is over (b,c)
                num = np.sum(T[a, :, :] * VW, axis=(0, 1))
                den = np.sum(VW * VW, axis=(0, 1))
                if den > 1e-12:
                    U[a, :] = num / den
                else:
                    U[a, :] = 0
            
            # Update V
            UW = np.einsum('ar,cr->ac r', U, W)  # shape (n*m, p*n, r)
            for b in range(m*p):
                num = np.sum(T[:, b, :] * UW, axis=(0, 2))
                den = np.sum(UW * UW, axis=(0, 2))
                if den > 1e-12:
                    V[b, :] = num / den
                else:
                    V[b, :] = 0
            
            # Update W
            UV = np.einsum('ar,br->ab r', U, V)  # shape (n*m, m*p, r)
            for c in range(p*n):
                num = np.sum(T[:, :, c] * UV, axis=(0, 1))
                den = np.sum(UV * UV, axis=(0, 1))
                if den > 1e-12:
                    W[c, :] = num / den
                else:
                    W[c, :] = 0
            
            # Check convergence
            if it % 50 == 0:
                current = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T - current)
                if err < 1e-10:
                    return U, V, W
        
        # Final check
        current = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(current, T, atol=1e-10):
            return U, V, W
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: the tensor has at least max(n*m, m*p, p*n) = 18 "nonzero" slices
    # But tensor rank can be higher. Let's try from 20 up.
    
    for r in range(20, min(rank_schoolbook + 1, 55)):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=200, seed=r)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r, T):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid decomposition at this rank
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(arr):
        """Convert numpy array to list of lists, rounding to nearest half-integer if close."""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-8:
                    new_row.append(rounded)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
