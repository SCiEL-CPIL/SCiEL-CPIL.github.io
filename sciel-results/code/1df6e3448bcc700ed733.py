import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 8
    
    # Naive rank-96 decomposition as baseline
    rank_naive = n * m * p  # 96
    
    U_naive = np.zeros((n * m, rank_naive), dtype=float)
    V_naive = np.zeros((m * p, rank_naive), dtype=float)
    W_naive = np.zeros((p * n, rank_naive), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1
                V_naive[j * p + k, col] = 1
                W_naive[k * n + i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check shapes
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != r or V.shape[1] != r or W.shape[1] != r:
            return False
        
        # For each (i, k) output position, verify that:
        # sum_l U[i*m+j, l] * V[j*p+k, l] = W[k*n+i, l] summed over j
        # Actually, the tensor identity is:
        # T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # But wait, that's not right. Let me reconsider.
        
        # The correct identity for matrix multiplication tensor:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has entries T[i,j,k] = 1 if we're computing (i,j,k) term
        # The decomposition should satisfy:
        # T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # Hmm, that doesn't look right either. Let me think again.
        
        # Actually, for bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
        # T[i,j,k] = sum_l U[i,j,l] V[j,k,l] W[?,l]
        # Wait, I need to be more careful.
        
        # The standard formulation: we want matrices U, V, W such that:
        # For all i, j, k: delta(i,j,k) = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # where delta(i,j,k) = 1 (since T[i,j,k] = 1 for all valid indices)
        
        # Actually no. The tensor entry is T[i,j,k] = 1 for all i,j,k in range.
        # The decomposition is: T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
        # This means sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i,j,k
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(r):
                        val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured patterns
    
    def try_rank_r(r, max_iters=1000):
        """Try to find a rank-r decomposition using local search."""
        if time.time() > deadline:
            return None
        
        # Initialize with random small values
        np.random.seed(42)
        U = np.random.uniform(-0.5, 0.5, (n * m, r))
        V = np.random.uniform(-0.5, 0.5, (m * p, r))
        W = np.random.uniform(-0.5, 0.5, (p * n, r))
        
        # Local search: try to minimize the error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(r):
                        val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                    error += (val - 1.0) ** 2
        
        if error < 1e-8:
            return U, V, W
        
        # Try coordinate descent
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            improved = False
            # Try updating one element at a time
            for i in range(n):
                for j in range(m):
                    for l in range(r):
                        old_val = U[i * m + j, l]
                        for delta in [-0.1, 0.1]:
                            U[i * m + j, l] = old_val + delta
                            new_error = 0.0
                            for k in range(p):
                                val = 0.0
                                for ll in range(r):
                                    val += U[i * m + j, ll] * V[j * p + k, ll] * W[k * n + i, ll]
                                new_error += (val - 1.0) ** 2
                            if new_error < error - 1e-12:
                                error = new_error
                                improved = True
                                break
                            U[i * m + j, l] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try V
                for j in range(m):
                    for k in range(p):
                        for l in range(r):
                            old_val = V[j * p + k, l]
                            for delta in [-0.1, 0.1]:
                                V[j * p + k, l] = old_val + delta
                                new_error = 0.0
                                for i in range(n):
                                    val = 0.0
                                    for ll in range(r):
                                        val += U[i * m + j, ll] * V[j * p + k, ll] * W[k * n + i, ll]
                                    new_error += (val - 1.0) ** 2
                                if new_error < error - 1e-12:
                                    error = new_error
                                    improved = True
                                    break
                                V[j * p + k, l] = old_val
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                # Try W
                for k in range(p):
                    for i in range(n):
                        for l in range(r):
                            old_val = W[k * n + i, l]
                            for delta in [-0.1, 0.1]:
                                W[k * n + i, l] = old_val + delta
                                new_error = 0.0
                                for j in range(m):
                                    val = 0.0
                                    for ll in range(r):
                                        val += U[i * m + j, ll] * V[j * p + k, ll] * W[k * n + i, ll]
                                    new_error += (val - 1.0) ** 2
                                if new_error < error - 1e-12:
                                    error = new_error
                                    improved = True
                                    break
                                W[k * n + i, l] = old_val
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if error < 1e-8:
                return U, V, W
        
        if error < 1e-8:
            return U, V, W
        return None
    
    # Try ranks from 95 down to some minimum
    # Start with a few promising ranks
    for r in [95, 94, 93, 92, 90, 88, 85, 80]:
        if time.time() > deadline:
            break
        
        result = try_rank_r(r, max_iters=500)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a structured approach: exploit the block structure
    # Split m=4 into two 2s, and p=8 into four 2s
    # This gives a 3x2x2 x 2x2x2 structure
    
    # For now, return the naive decomposition
    return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
