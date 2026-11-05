import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    # Tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Since we are multiplying 3x3 by 3x7, the tensor is actually the identity mapping?
    # No, the "matrix multiplication tensor" T such that (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T_{i,j,k} is 1 if i,j,k are indices, 0 otherwise?
    # Wait, the standard tensor for matrix multiplication is T_{i,j,k} = delta_{i,j,k}? No.
    # Let's check the trace convention.
    # W[k*n+i, r]
    # U[i*m+j, r]
    # V[j*p+k, r]
    # Product sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This looks like the standard multiplication tensor where T_{i,j,k} = 1 if the indices match?
    # Actually, for matrix multiplication, the tensor T_{i,j,k} is 1 if i=j=k? No.
    # The equation is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor entry T_{i,j,k} corresponds to the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # This coefficient is 1 if the indices are valid, 0 otherwise?
    # In the tensor network form, T_{i,j,k} is the structural tensor.
    # For matrix multiplication, T_{i,j,k} = 1 for all valid i,j,k?
    # No, T_{i,j,k} = 1 if the operation is A_{i,j} * B_{j,k} -> C_{i,k}.
    # So yes, T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, is it?
    # If T_{i,j,k} = 1, then the decomposition is sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1.
    # This is exactly the rank of the matrix multiplication tensor.
    
    # Let's verify with the schoolbook fallback in the prompt.
    # U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1.
    # This creates a term 1*1*1 = 1 for each (i,j,k) combination.
    # So yes, we are decomposing the all-ones tensor of size n x m x p.
    
    # Known results for 3x3x3 multiplication:
    # Schoolbook: 27
    # Strassen: 7 (for 2x2x2), but for 3x3x3?
    # The rank of 3x3x3 matrix multiplication is known to be 21?
    # Actually, the rank of multiplying 3x3 by 3x3 is 21?
    # Let's check literature.
    # The bilinear complexity of multiplying 3x3 matrices is 21?
    # No, it is known that the rank is at most 21.
    # Wait, 3x3x3 rank is 21?
    # Let's check 2x2x2: rank 7.
    # 3x3x3: rank 21 is a known bound?
    # Actually, I recall that the rank of 3x3 matrix multiplication is 21?
    # Let's try to find a decomposition of rank 21 for 3x3x3.
    
    # For 3x3x7, we can split the 7 columns into 3+3+1.
    # 3x3x3 (rank 21) + 3x3x3 (rank 21) + 3x3x1 (rank 9)?
    # Total rank 21+21+9 = 51.
    # Can we do better?
    # The rank of 3x3x7 multiplication.
    # Lower bound: (n+m+p-2)^2? No.
    # Lower bound for M_{n,m,p} is max(nm, mp, pn)?
    # nm=9, mp=21, pn=21.
    # So rank >= 21.
    
    # Is it possible to get rank 21?
    # If we could get rank 21, that would be optimal.
    # But usually, the rank is higher.
    # Let's try to search for a low rank decomposition using alternating least squares.
    
    # Strategy:
    # 1. Try ranks from 21 upwards.
    # 2. For each rank, initialize U, V, W randomly.
    # 3. Perform alternating minimization (ALS) to solve for U, V, W.
    # 4. Check if the decomposition is exact.
    # 5. If found, return it.
    
    # ALS Step:
    # To solve for U:
    # For each (i,j), we have equation: sum_k T[i,j,k] = sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r] / something?)
    # Actually, the tensor equation is:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For fixed V and W, we can solve for U.
    # For each (i,j), the vector of equations over k is:
    # [T[i,j,0], T[i,j,1], ..., T[i,j,p-1]] = sum_r U[i*m+j, r] * [V[j*p+0, r] W[0*n+i, r], V[j*p+1, r] W[1*n+i, r], ...]
    # Let M_VW[j, k, r] = V[j*p+k, r] * W[k*n+i, r]?
    # No, W index depends on i.
    # So for fixed i, we have a matrix M_VW_j (size p x rank) where M_VW_j[k, r] = V[j*p+k, r] * W[k*n+i, r].
    # Then we want U[i*m+j, :] such that M_VW_j @ U[i*m+j, :].T = T[i,j, :].
    # This is a linear system: (M_VW_j^T M_VW_j) U[i*m+j, :].T = M_VW_j^T T[i,j, :].
    # We can solve this least squares problem.
    
    # Similarly for V and W.
    
    # Let's implement this.
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Target rank range
    # Lower bound is 21. Upper bound is 63 (schoolbook).
    # Let's try 21 to 35 first.
    
    max_rank_to_try = 35
    min_rank_to_try = 21
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + rank * 100 + attempt)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            for iteration in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # Build M_VW_j (p x rank)
                        M_VW = np.zeros((p, rank))
                        for k in range(p):
                            M_VW[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        # Solve M_VW @ u_j = T[i,j,:]
                        # u_j is length rank
                        # M_VW^T M_VW u_j = M_VW^T T[i,j,:]
                        A_mat = M_VW.T @ M_VW
                        b_vec = M_VW.T @ np.ones(p)  # T[i,j,k] = 1
                        # Add small regularization for stability
                        A_mat += 1e-10 * np.eye(rank)
                        try:
                            U[i*m + j, :] = np.linalg.solve(A_mat, b_vec)
                        except np.linalg.LinAlgError:
                            U[i*m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        M_UW = np.zeros((n, rank))
                        for i in range(n):
                            M_UW[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        A_mat = M_UW.T @ M_UW
                        b_vec = M_UW.T @ np.ones(n)
                        A_mat += 1e-10 * np.eye(rank)
                        try:
                            V[j*p + k, :] = np.linalg.solve(A_mat, b_vec)
                        except np.linalg.LinAlgError:
                            V[j*p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        M_UV = np.zeros((m, rank))
                        for j in range(m):
                            M_UV[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        A_mat = M_UV.T @ M_UV
                        b_vec = M_UV.T @ np.ones(m)
                        A_mat += 1e-10 * np.eye(rank)
                        try:
                            W[k*n + i, :] = np.linalg.solve(A_mat, b_vec)
                        except np.linalg.LinAlgError:
                            W[k*n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                # Check convergence / validity
                # Compute the reconstructed tensor
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-6:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
                
                if time.time() > deadline:
                    break
        
        if best_rank is not None:
            break
    
    # If no low rank found, fallback to schoolbook
    if best_rank is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        best_U = U
        best_V = V
        best_W = W
    
    # Convert to lists of lists of floats (or integers if close)
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
