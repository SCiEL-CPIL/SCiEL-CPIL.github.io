"""Matrix multiplication 2,5,6 rank minimization via structured block decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy:
    1. Build the tensor T in indices (i, j, k) where:
       - i in [0,2): output row
       - j in [0,5): shared index (col of A = row of B)
       - k in [0,6): output col
       T[i,j,k] = 1 if the output element (i,k) depends on input element (j)
       Actually, the matrix multiplication tensor is:
       C[i,k] = sum_j A[i,j] * B[j,k]
       The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
       So T[i,j,k] = 1 for all i,j,k.
       
       Wait, that's not right. The tensor for matrix multiplication M_{n,m} * M_{m,p} is:
       T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} ⊗ F_{j,k} ⊗ G_{i,k}
       where E_{i,j} has 1 at (i,j), F_{j,k} has 1 at (j,k), G_{i,k} has 1 at (i,k).
       
       In the standard convention for the interface:
       U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
       
       The decomposition is:
       sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
       
       Where T[i*m+j, j*p+k, k*n+i] = 1 (for the multiplication tensor)
       
       Actually, let me re-read the interface:
       U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
       
       The tensor T has entries T[a, b, c] where:
       - a indexes into n*m (so a = i*m + j, i in [0,n), j in [0,m))
       - b indexes into m*p (so b = j*p + k, j in [0,m), k in [0,p))
       - c indexes into p*n (so c = k*n + i, k in [0,p), i in [0,n))
       
       The multiplication tensor satisfies:
       T[i*m+j, j*p+k, k*n+i] = 1 if the same j appears in both a and b indices
       Wait, no. Let me think again.
       
       For matrix multiplication C = A * B where A is n×m and B is m×p:
       C[i,k] = sum_j A[i,j] * B[j,k]
       
       The tensor is: T = sum_{i,j,k} (E_{i,j} ⊗ F_{j,k} ⊗ G_{i,k})
       
       In flattened indices:
       - A index: i*m + j
       - B index: j*p + k  
       - C index: k*n + i (per the problem statement W[k*n+i, r])
       
       So T[i*m+j, j*p+k, k*n+i] = 1
       
       The schoolbook decomposition has rank n*m*p with one term per (i,j,k):
       U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
       
       Now, for 2x5x6, can we do better than 60?
       
       Key insight: The tensor has a block structure. 
       For fixed j, we have a 2x6 "slice" that's rank 2 (since C[:,k] = A[:,j] * B[j,k]).
       
       Actually, the tensor can be viewed as:
       T = sum_j (A_j ⊗ B_j ⊗ C_j) where for each j:
       - A_j is a vector in R^{n*m} with 1's at positions i*m+j for i in [0,n)
       - B_j is a vector in R^{m*p} with 1's at positions j*p+k for k in [0,p)
       - C_j is a tensor in R^{p*n}... 
       
       Hmm, this is getting complicated. Let me try a different approach.
       
       For small cases, the rank is known:
       - R(2,2,2) = 7 (Strassen)
       - R(2,2,1) = 2*2 = 4? No, it's actually 2*2=4 for multiplying 2x2 by 2x1, but that's just 4 multiplies.
       
       For R(n, m, p) with one dimension small:
       - R(1, m, p) = m*p (just dot products)
       - R(2, 1, p) = 2*p? Or can we do better?
       
       Actually, R(2, 1, p): C[i,k] = A[i] * B[k] for i in [0,2), k in [0,p).
       This is an outer product, so rank = 2*p.
       
       R(2, m, 1): C[i] = sum_j A[i,j] * B[j] for i in [0,2).
       This is 2 independent dot products of length m, so rank = 2*m.
       
       For R(2, m, p) with m, p > 1, the rank is generally less than 2*m*p.
       
       Known result: R(2, m, m) <= 2*m^2 - m for some values? 
       
       Let me try to use the fact that n=2 is small.
       
       For n=2, we can write:
       C[0,k] = sum_j A[0,j] * B[j,k]
       C[1,k] = sum_j A[1,j] * B[j,k]
       
       The tensor is: T = sum_j (A_j ⊗ B_j ⊗ C_j)
       where A_j has 1's at (0*m+j) and (1*m+j), 
             B_j has 1's at (j*p+0), (j*p+1), ..., (j*p+p-1),
             C_j has 1's at (k*2+0) and (k*2+1) for k in [0,p).
       
       Actually, C_j is a matrix in R^{p*2} with entries C_j[k*2+i] = 1 for all k,i.
       So C_j is the all-ones vector of length 2p.
       
       Wait, that's not right either. Let me be more careful.
       
       T[i*m+j, j*p+k, k*n+i] = 1
       
       For n=2, m=5, p=6:
       T[i*5+j, j*6+k, k*2+i] = 1 for i in [0,2), j in [0,5), k in [0,6).
       
       The schoolbook has 60 terms. Can we group them?
       
       Observation: For fixed j, the term involves:
       - U positions: i*5+j for i=0,1
       - V positions: j*6+k for k=0..5
       - W positions: k*2+i for k=0..5, i=0,1
       
       So for fixed j, we have 2*6 = 12 (i,k) pairs, each giving a term.
       
       Can we decompose the 2x6 sub-tensor for fixed j into fewer than 12 rank-1 terms?
       The sub-tensor for fixed j is: S_j[i,k] = 1 for all i in [0,2), k in [0,6).
       This is a 2x6 all-ones matrix, which has rank 2 (as a matrix).
       But as a tensor with the specific index structure, the "rank" might be different.
       
       Actually, I think I need to approach this differently. Let me try to use random search or algebraic methods to find a lower-rank decomposition.
       
       Given the time constraints, let me try a structured approach:
       1. Start with the schoolbook decomposition (rank 60).
       2. Try to find linear dependencies among the rank-1 tensors to reduce the rank.
       
       Two rank-1 tensors u⊗v⊗w and u'⊗v'⊗w' are linearly dependent if they are scalar multiples.
       In the schoolbook, all terms are distinct, so there are no obvious dependencies.
       
       However, we might be able to find a different set of rank-1 tensors that spans the same space.
       
       Let me try a random optimization approach:
       - Parameterize a rank-r decomposition with r < 60.
       - Use random initialization and gradient descent to minimize the error.
       - Check if we can achieve zero error.
       
       Given the 120s limit, this might be feasible for r close to 60.
       
       Actually, let me try a simpler approach: use the known Strassen-like tricks.
       
       For 2x2x2, Strassen reduces 8 to 7. The key is that the tensor has a non-trivial structure that allows one multiplication to be saved.
       
       For 2x5x6, maybe we can save some multiplications similarly.
       
       Let me try to use a block decomposition where we group the j indices.
       
       Actually, I'll implement a random search that tries to find a low-rank decomposition.
       """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the tensor for verification
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T
    
    def verify(U, V, W, T, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # Reconstruct tensor
        T_rec = np.zeros_like(T)
        for c in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T, atol=tol)
    
    T = build_tensor()
    
    # Try random search for lower rank
    best_result = schoolbook()
    
    # Try ranks from 55 down to 40
    for rank in range(55, 39, -1):
        if _time_remaining() < 30:
            break
            
        # Multiple random trials
        max_trials = 10
        for trial in range(max_trials):
            if _time_remaining() < 15:
                    break
                    
            # Initialize with random values
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Try to fit using alternating least squares
            converged = False
            for it in range(50):
                if _time_remaining() < 10:
                    break
                    
                # Update U
                # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # For fixed c, U[:,c] should be such that U[:,c] ⊗ V[:,c] ⊗ W[:,c] approximates T
                # This is a tensor regression problem.
                
                # Reshape T to (n*m, m*p*p*n)
                T_mat = T.reshape(n * m, -1)
                
                # For each row a of U, we want: sum_c U[a,c] * (V[:,c] ⊗ W[:,c])[a] ≈ T_mat[a,:]
                # This is a linear system in U[:,c]
                
                # Build the design matrix
                # M[a, c] = ||V[:,c] ⊗ W[:,c]||^2 ... no, that's not right.
                
                # Actually, the tensor entry T[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # For fixed a, T[a, b, d] = sum_c U[a,c] * (V[b,c] * W[d,c])
                # So if we define X_c[b,d] = V[b,c] * W[d,c], then T[a] = sum_c U[a,c] * X_c
                # where T[a] is the slice T[a, :, :] flattened.
                
                T_a = T.reshape(n * m, m * p * p * n)
                
                # Build design matrix D of shape (m*p*p*n, rank)
                D = np.zeros((m * p * p * n, rank))
                for c in range(rank):
                    D[:, c] = np.outer(V[:, c], W[:, c]).reshape(-1)
                
                # Solve U D = T_a, so U = T_a D^+
                try:
                    U_new = np.linalg.lstsq(D, T_a.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                T_b = T.reshape(m * p, n * m * p * n)
                D_v = np.zeros((n * m * p * n, rank))
                for c in range(rank):
                    D_v[:, c] = np.outer(U_new[:, c], W[:, c]).reshape(-1)
                try:
                    V_new = np.linalg.lstsq(D_v, T_b.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                T_d = T.reshape(p * n, n * m * m * p)
                D_w = np.zeros((n * m * m * p, rank))
                for c in range(rank):
                    D_w[:, c] = np.outer(U_new[:, c], V_new[:, c]).reshape(-1)
                try:
                    W_new = np.linalg.lstsq(D_w, T_d.T, rcond=None)[0].T
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if it % 5 == 0:
                    if verify(U, V, W, T, tol=1e-8):
                        converged = True
                        break
            
            if converged:
                # Found a valid decomposition of this rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if _time_remaining() < 20:
            break
    
    return best_result
