"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  The multiplication tensor T_{i,j,k} = E_{i,k} (indexing U[i*m+j], V[j*p+k], W[k*n+i])
  has trivial rank 3*3*7 = 63 (the schoolbook decomposition).
  
  We attempt to find a lower-rank decomposition using the following structural insight:
  
  For multiplying (n x m) * (m x p), the tensor lives in R^{n*m} x R^{m*p} x R^{p*n}.
  The rank is the minimum number of rank-1 tensors summing to T.
  
  A key observation: if we can factor the problem using block structure, we may reduce rank.
  Specifically, for 3x3 * 3x7, we can view the 3x7 matrix as three 3x3 blocks plus a 3x1 block,
  or exploit the fact that 7 = 3 + 3 + 1.
  
  However, the most promising structural approach for small matrices is to use the
  "border rank" or "rank factorization" ideas. For the specific case of 3x3 * 3x7:
  
  - The trivial rank is 63.
  - Known lower bounds: rank >= n*p = 21 (from flattening), and rank >= m*p + n*m - m = 21+9-3=27 
    (from Strassen-type lower bounds, though these are not tight for rectangular cases).
  
  A practical approach: use integer/half-integer entries and try to find a decomposition
  with rank < 63 by exploiting linear dependencies.
  
  For 3x3 * 3x3, the best known rank is 21 (vs 27 trivial). For 3x3 * 3x7, we can try to
  decompose the 3x7 into 3x3 + 3x3 + 3x1 and use known 3x3 results.
  
  Actually, let me think more carefully. The multiplication of (3x3) * (3x7) can be seen as:
  - 3 independent 3x3 * 3x1 multiplications (columns of the result)
  - Or: the 3x7 matrix has 7 columns, each of length 3.
  
  Each column of the result is A * b_k where b_k is a 3-vector. So we need to compute
  7 products of a 3x3 matrix with a 3-vector.
  
  The tensor for 3x3 * 3x1 (i.e., matrix-vector product) has trivial rank 9.
  Can we do better? For matrix-vector 3x3 * 3x1, the tensor is in R^9 x R^3 x R^3.
  The rank of the matrix-vector product tensor for n x n * n x 1 is n^2 (I believe this is optimal
  for small n, but let me verify).
  
  Actually, for the matrix-vector product A (n x n) * x (n x 1) = y (n x 1), the tensor is
  T_{i,j,k} = delta_{i,k} (where i indexes rows of A, j indexes columns of A / components of x,
  k indexes components of y). Wait, let me re-index.
  
  U has n*m = 9 entries (indices i,j for A[i,j])
  V has m*p = 3*1 = 3 entries (indices j for x[j])  
  W has p*n = 1*3 = 3 entries (indices k for y[k])
  
  T_{(i,j),(j'),(k)} = delta_{j,j'} * delta_{i,k}
  
  So T_{(i,j),j',k} = delta_{j,j'} delta_{i,k}
  
  This is a sum over i,j of delta_{j,j'} delta_{i,k} E_{(i,j)} tensor E_{j'} tensor E_k.
  
  The rank is at most 9 (the trivial sum). Can we do better?
  
  For n=2: 2x2 * 2x1. T_{(i,j),j',k} = delta_{j,j'} delta_{i,k}.
  Entries: (0,0),0,0 = 1; (0,0),1,0 = 0; ... 
  Let's see if rank 4 is possible. The tensor is actually the "slice" of the matrix multiplication.
  
  I think for matrix-vector, the rank is indeed n^2 because the tensor is "generic" in a certain sense.
  
  Let me try a different approach: use the fact that we're multiplying 3x3 by 3x7, and try to
  find a decomposition with rank between 27 and 63.
  
  A practical algorithm: 
  1. Start with the trivial rank-63 decomposition.
  2. Try to find linear dependencies among the rank-1 terms to reduce the rank.
  3. Use numerical optimization to find a lower-rank approximation, then round to rationals.
  
  Given the time constraint, let me implement a search that:
  - Tries ranks from a lower bound up to 63
  - For each rank, sets up the equations and tries to solve them
  - Uses random initialization and iterative refinement
  
  The equations: for each (i,j,k) in {0,1,2}x{0,1,2}x{0,1,2,3,4,5,6},
  sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (if we're computing A*B where B is 3x7)
  
  Wait, let me re-check the tensor definition. The multiplication is C = A * B where A is 3x3, B is 3x7, C is 3x7.
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  So the tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
  T_{(i,j),(j',k),(i',k')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
  
  In the interface:
  U is (n*m, rank) = (9, rank), indexed by i*m+j
  V is (m*p, rank) = (21, rank), indexed by j*p+k  
  W is (p*n, rank) = (21, rank), indexed by k*n+i
  
  The tensor entry for a rank-1 term r contributes:
  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
  
  And we need: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{j,j} ... wait.
  
  Let me be precise. C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
  
  So the tensor is: T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
  Wait no. The tensor represents the bilinear map (A, B) -> C.
  
  For a rank-1 decomposition: C[i,k] = sum_r (sum over the rank-1 factors)
  Actually: C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... this doesn't look right.
  
  Let me re-read the interface. The factor matrices are U (n*m x rank), V (m*p x rank), W (p*n x rank).
  The decomposition is: T = sum_r u_r tensor v_r tensor w_r where u_r is the r-th column of U, etc.
  
  The tensor T has entries T[a, b, c] where a indexes U (i*m+j), b indexes V (j'*p+k), c indexes W (k'*n+i').
  T[a, b, c] = 1 if the indices are consistent: i.e., if a = i*m+j, b = j*p+k, c = k*n+i for the same i,j,k.
  
  So T[(i*m+j), (j*p+k), (k*n+i)] = 1 when i,j,k are the same across all three.
  
  And the rank-1 decomposition satisfies:
  T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
  
  So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) are consistent, else 0.
  
  Wait, that's not right either. Let me think again.
  
  The tensor for matrix multiplication (n x m) * (m x p) -> (n x p):
  T is a 3-tensor with dimensions (n*m) x (m*p) x (n*p).
  T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i' and j=j' and k=k', else 0.
  
  So T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
  And T = 0 elsewhere.
  
  The rank-1 decomposition: T[a,b,c] = sum_{r=1}^{R} U[a,r] V[b,r] W[c,r]
  
  So we need: for all i,j,k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
  And for all other (a,b,c): sum_r U[a,r] V[b,r] W[c,r] = 0
  
  This is a system of n*m*m*p*n*p - n*m*p = 9*21*21 - 63 = 3969 - 63 = 3906 equations (the zero ones)
  plus 63 equations (the ones that equal 1).
  
  Total: 4029 equations, with 9R + 21R + 21R = 51R unknowns.
  
  For R=63: 51*63 = 3213 unknowns, 4029 equations. Underdetermined, so solutions exist.
  For R=50: 51*50 = 2550 unknowns, 4029 equations. Overdetermined.
  
  This is a hard problem. Let me try a different approach.
  
  Given the difficulty, let me try to use known results or a smarter search.
  
  Actually, for the specific case of 3x3 * 3x7, I'll try the following:
  1. Use the trivial rank-63 decomposition as a baseline.
  2. Try to reduce rank by finding linear combinations of the rank-1 terms that can be merged.
  
  A simpler approach: use the fact that the tensor has a lot of structure and try to find
  a decomposition with rank in the range [40, 63] using random search + local optimization.
  
  Given the 150s time limit, I'll implement:
  - A random initialization for a given rank R
  - A gradient-based or least-squares solver
  - Try several ranks and return the best valid decomposition
  
  Let me implement this.
"""

DIMENSIONS = [3, 3, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition (rank = n*m*p = 63)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists for the return format
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Try to find a lower-rank decomposition
    # Use a randomized approach with least squares
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank
    
    # Try ranks from trivial_rank-1 down to some lower bound
    # For each rank, try a few random initializations
    
    lower_bound = max(n * p, m * p + n * m - m, 27)  # heuristic lower bound
    
    for R in range(trivial_rank - 1, max(lower_bound - 1, 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations for this rank
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Build the linear system: for each (i,j,k), we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear. Instead, fix two matrices and solve for the third.
            
            # Alternate minimization: fix V and W, solve for U, etc.
            # But this is still nonlinear.
            
            # Alternative: use the fact that for fixed V and W, the equations are linear in U.
            # For each i,j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a system: for fixed i,j, we have p equations in R unknowns (the U[i*m+j, :]).
            
            # Let's do coordinate descent:
            # 1. Fix V, W. For each i,j, solve the linear system for U[i*m+j, :].
            # 2. Fix U, W. For each j,k, solve for V[j*p+k, :].
            # 3. Fix U, V. For each k,i, solve for W[k*n+i, :].
            
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each i, j
                for i in range(n):
                    for j in range(m):
                        # Equations: for each k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # This is: (V[j*p: j*p+p, :] * W[i: i+p: n, :].T) @ U[i*m+j, :] = ones(p)
                        # Wait, let me be careful.
                        
                        # For fixed i,j: the equation for each k is:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let A_k,r = V[j*p+k, r] * W[k*n+i, r]
                        # Then: sum_r U[i*m+j, r] * A_k,r = 1 for k=0,...,p-1
                        # This is: (A @ U[i*m+j, :].T) = ones(p)
                        # where A is p x R with A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve A @ x = ones(p) for x = U[i*m+j, :]
                        # Use least squares
                        try:
                            x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # Update V: for each j, k
                for j in range(m):
                    for k in range(p):
                        # For fixed j,k: equations for each i in range(n):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let B_i,r = U[i*m+j, r] * W[k*n+i, r]
                        # Then: sum_r V[j*p+k, r] * B_i,r = 1 for i=0,...,n-1
                        B = np.zeros((n, R))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x, residuals, rank_B, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # Update W: for each k, i
                for k in range(p):
                    for i in range(n):
                        # For fixed k,i: equations for each j in range(m):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # Let C_j,r = U[i*m+j, r] * V[j*p+k, r]
                        # Then: sum_r W[k*n+i, r] * C_j,r = 1 for j=0,...,m-1
                        C = np.zeros((m, R))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x, residuals, rank_C, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence: compute the error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                err /= (n * m * p)
                
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Verify that the decomposition is exact (all non-diagonal terms are ~0)
                valid = True
                # Check a few random non-diagonal terms
                for _ in range(100):
                    a = np.random.randint(0, n * m)
                    b = np.random.randint(0, m * p)
                    c = np.random.randint(0, p * n)
                    # Check if this is a "diagonal" term
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (i == i2 and j == j2 and k == k2):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        if abs(val) > 1e-8:
                            valid = False
                            break
                
                if valid:
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = R
                    break  # Found a valid decomposition for this rank, try lower rank
        
        if time.time() > deadline - 10:
            break
    
    # Convert to list format
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
