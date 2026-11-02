"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools
import numpy as np
from scipy.linalg import svd, qr

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0
        # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor for multiplication has T[i,j,k] = 1 for all i,j,k (it's the full product tensor)
        # Actually, the multiplication tensor M_{i,j,k} = e_i^T A e_j * e_j^T B e_k = A_{i,j} B_{j,k}
        # The rank-r decomposition means: M_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        # where M_{i,j,k} = 1 (the structure tensor of matrix mult)
        
        # Check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
        # (since the multiplication tensor has all entries = 1 in the appropriate structure)
        
        # Actually, let me reconsider. The matrix multiplication tensor T has:
        # T[i,j,k] = 1 for the "structural" entries
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify by computing the sum for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_random_search(max_rank, time_limit):
        """Try to find a low-rank decomposition via random search."""
        best = None
        start = time.time()
        
        for rank in range(max_rank, 83, -1):  # 83 = 3*4*7, try from high to low
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 10:
                break
            
            # Try several random initializations
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Random integer/half-integer entries
                U = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(n*m, rank))
                V = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(m*p, rank))
                W = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(p*n, rank))
                
                # Quick check using tensor contraction
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # We want T[i,j,k] = 1 for all i,j,k
                
                # Efficient check: compute the full tensor
                # U reshaped: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
                # This is a bit tricky to vectorize directly due to the index permutations
                # Let's do it more carefully
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                            if abs(s - 1.0) > 1e-4:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    best = (rank, U.tolist(), V.tolist(), W.tolist())
                    return best
        
        return best
    
    def svd_based_search(time_limit):
        """Use SVD/QR to identify structure, then refine."""
        start = time.time()
        
        # Build the multiplication tensor
        # T[i,j,k] = 1 for all i,j,k (structural tensor of matrix mult)
        # Flatten: T_flat has entries T[i*m+j, k] or similar
        
        # The tensor T of shape (n*m, m*p, p*n) where T[a, b, c] corresponds to
        # a = i*m+j, b = j*p+k, c = k*n+i, and T[a,b,c] = 1 if consistent, else 0
        
        # Actually for matrix multiplication, the tensor is:
        # M[i,j,k] = 1 (it's the all-ones tensor in the appropriate structure)
        # No wait - the matrix multiplication tensor T has T[i,j,k] = 1 iff the indices
        # form a valid "cycle" in the multiplication. For standard matrix mult:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k].
        
        # So T is just the all-ones tensor of shape (n, m, p)!
        # That can't be right for rank minimization...
        
        # Let me reconsider. The multiplication tensor is:
        # T = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ e_k^* where the output is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # Actually T[i,j,k] = 1 for all i,j,k in the standard formulation.
        
        # The rank of the all-ones tensor of shape (n,m,p) is... well, 
        # rank(ones(n,m,p)) = min(n*m, m*p, p*n) in general? No.
        # The all-ones tensor ones(n,m,p) has rank 1! Because ones(n,m,p) = vec(1) ⊗ vec(1) ⊗ vec(1)
        
        # Wait, that's not right either. Let me think again.
        # The matrix multiplication tensor for multiplying n×m by m×p matrices:
        # It's a tensor T of shape (n*m, m*p, p*n) where
        # T[(i,j), (j,k), (k,i)] = 1 and 0 otherwise.
        
        # So T is NOT all ones. T[a,b,c] = 1 only when the indices are consistent:
        # a = i*m+j, b = j'*p+k, c = k'*n+i' where j=j' and k=k' and i=i'.
        
        # Let me build it properly.
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Now we want to find the tensor rank of T.
        # Tensor rank = minimum r such that T[a,b,c] = sum_{r'=1}^{r} U[a,r'] V[b,r'] W[c,r']
        
        # Strategy: Try to find low-rank decompositions
        # First, get an upper bound via SVD of unfoldings
        
        # Unfold along first mode: T_1 has shape (n*m, m*p*p*n)
        # T_1[a, (b,c)] = T[a,b,c]
        T_1 = T.reshape(n*m, m*p*p*n)
        T_2 = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
        T_3 = T.transpose(2,0,1).reshape(p*n, n*m*m*p)
        
        # SVD of unfoldings gives lower bounds on rank
        U1, S1, Vt1 = svd(T_1, full_matrices=False)
        U2, S2, Vt2 = svd(T_2, full_matrices=False)
        U3, S3, Vt3 = svd(T_3, full_matrices=False)
        
        rank_lb1 = int(np.sum(S1 > 1e-10))
        rank_lb2 = int(np.sum(S2 > 1e-10))
        rank_lb3 = int(np.sum(S3 > 1e-10))
        rank_lb = max(rank_lb1, rank_lb2, rank_lb3)
        
        # Try ranks from lower bound up
        for rank in range(rank_lb, 84):
            if time.time() - start > time_limit or time.time() > deadline - 15:
                break
            
            # Try to find a rank-r decomposition
            # Use the SVD to get initial guesses
            if rank <= min(T_1.shape):
                # Take top-r components
                U_init = U1[:, :rank]
                # This gives a rank-r approximation in the first unfolding sense
                # But we need exact decomposition
            
            # Random search with structure
            for attempt in range(20):
                if time.time() > deadline - 15:
                    break
                
                # Initialize with random values, then try to solve
                U = np.random.randn(n*m, rank) * 0.5
                V = np.random.randn(m*p, rank) * 0.5
                W = np.random.randn(p*n, rank) * 0.5
                
                # Try to refine using alternating least squares (ALS)
                # For each r, fix U and V, solve for W:
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed U, V: W[c,r] = ?
                # This is a system of equations. For each (a,b), we have:
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] for all c
                # This is linear in W.
                
                # Actually, let's use a different approach:
                # Fix W, then for each c, we have a matrix equation:
                # T[:,:,c] = sum_r U[:,r] V[:,r]^T * W[c,r]
                # So T[:,:,c] = sum_r W[c,r] * (U[:,r] V[:,r]^T)
                
                # This is a linear system in the entries of U and V given W, or vice versa.
                # Let's do ALS:
                
                for iteration in range(50):
                    if time.time() > deadline - 15:
                        break
                    
                    # Update W: for each c, solve T[:,:,c] = sum_r W[c,r] * outer(U[:,r], V[:,r])
                    # Flatten: T_flat_c = sum_r W[c,r] * vec(outer(U[:,r], V[:,r]))
                    # This is a linear system: A_c * w_c = t_c where A_c[r] = vec(outer(U[:,r], V[:,r]))
                    
                    # But this might be overdetermined. Let's use least squares.
                    # Actually, let's just do gradient descent or a simpler approach.
                    
                    # Simpler: just check if current U,V,W works
                    # T_check[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    
                    pass  # Skip complex optimization for now
                
                break  # Skip to next rank
        
        return None  # This approach is too slow; fall back to schoolbook
    
    # First, try the schoolbook (known to work)
    best_result = schoolbook()
    best_rank = n * m * p  # 83
    
    # Try to improve with randomized search
    # The tensor rank of matrix multiplication for 3x4 by 4x7 is unknown,
    # but it's at most 83 (schoolbook) and at least max(3*4, 4*7, 7*3) = max(12,28,21) = 28
    
    # Try a limited search for lower ranks
    search_time = min(100, deadline - time.time() - 20)
    if search_time > 30:
        result = try_random_search(best_rank - 1, search_time)
        if result is not None:
            rank, U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
