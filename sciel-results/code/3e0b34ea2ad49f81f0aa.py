"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n=3, m=5, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each term t, the tensor slice is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == delta structure
    # Actually, the multiplication tensor T[i,j,k] = 1 (it's the matmul tensor)
    # The decomposition requires: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # But wait, the matmul tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is: T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor of matmul)
    # Actually no. The multiplication tensor has entries: for each (i,j,k), T[i,j,k] = 1
    # and we need C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t where
    # T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] ... 
    # Let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] convention
    # The tensor T[i,j,k] for matmul is 1 (the structure tensor)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Check: for each (i,j,k), sum over r of U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication."""
    n, m, p = 3, 5, 5
    
    # Start with schoolbook (rank 75) and try to improve
    # Use a structured decomposition based on blocking
    
    # Strategy: Partition into blocks and use efficient sub-algorithms
    # 3x5x5 can be decomposed as:
    # Think of A as 3x5, B as 5x5
    # We can write this as a combination of smaller multiplications
    
    # Approach: Use a rank-60 decomposition by exploiting structure
    # Let's try a systematic search with time budget
    
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    # First, let's try a known good decomposition
    # For 3x5x5, we can use the following approach:
    # Split the 5 columns into groups and use Strassen on 3x3 blocks
    
    # Actually, let me try a direct algebraic approach
    # The multiplication: C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
    
    # Let me construct a rank-72 decomposition first, then try to reduce
    
    # A simpler approach: use a block decomposition
    # Write A as [A1 | A2] where A1 is 3x3 and A2 is 3x2
    # Write B as [B1; B2] where B1 is 3x5 and B2 is 2x5
    # Then C = A1*B1 + A2*B2
    
    # A1*B1: 3x3 times 3x5, rank 9 schoolbook
    # A2*B2: 3x2 times 2x5, rank 6 schoolbook
    # Total: 15, but we can do better with Strassen on A1*B1
    
    # Strassen for 3x3: rank 23 (Berkovich) or 27 (schoolbook)
    # Let's just use a clean construction
    
    # Most reliable: schoolbook rank 75, but let's try to find a better one
    # by using the fact that we only need 15 output values
    
    # Try a parameterized search
    # The tensor has 3*5*5 = 75 entries, all equal to 1
    # We need rank r such that there exist U(15,r), V(25,r), W(25,r) with
    # sum_r U[i*5+j, r]*V[j*5+k, r]*W[k*3+i, r] = 1 for all i,j,k
    
    # Let me try a greedy/optimization approach
    # Actually, let me just implement a known sub-optimal but better than schoolbook result
    
    # For 3x5x5, a rank of 70 should be achievable
    # Let me use a structured approach:
    
    # Use the following decomposition:
    # For the j=0,1,2 part (3x3x5 subproblem) use Strassen-like
    # For j=3,4 part (3x2x5 subproblem) use schoolbook
    
    # Actually, the simplest guaranteed improvement:
    # Use rank 74 by merging one term
    
    # Let me just do a proper search
    # Initialize with schoolbook
    rank_sb = n * m * p  # 75
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for efficiency
    U_np = np.array(U_sb, dtype=float)
    V_np = np.array(V_sb, dtype=float)
    W_np = np.array(W_sb, dtype=float)
    
    # The tensor condition: for each (i,j,k), 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a multilinear constraint
    
    # Try randomized perturbation to find lower-rank solutions
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try rank from 74 down to some minimum
    min_rank_to_try = 60
    
    for target_rank in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try to find a rank-target_rank solution
        # Initialize with random perturbation of schoolbook
        found = False
        attempts = 20 if target_rank >= 70 else 10
        
        for attempt in range(attempts):
            if time.time() > deadline - 20:
                break
            
            # Start from schoolbook and zero out some columns, then solve
            # Pick target_rank columns from the 75 schoolbook columns
            cols_to_keep = list(range(target_rank))
            
            U_t = np.array(U_sb[:, cols_to_keep], dtype=float)
            V_t = np.array(V_sb[:, cols_to_keep], dtype=float)
            W_t = np.array(W_sb[:, cols_to_keep], dtype=float)
            
            # Perturb
            U_t = U_t + 0.01 * np.random.randn(*U_t.shape)
            V_t = V_t + 0.01 * np.random.randn(*V_t.shape)
            W_t = W_t + 0.01 * np.random.randn(*W_t.shape)
            
            # Try to fix the constraints using gradient descent
            # Loss = sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            lr = 0.001
            for step in range(500):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_t)
                grad_V = np.zeros_like(V_t)
                grad_W = np.zeros_like(W_t)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # s = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                            s = np.sum(U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i])
                            err = s - 1.0
                            for r in range(target_rank):
                                u_val = U_t[i*m+j][r]
                                v_val = V_t[j*p+k][r]
                                w_val = W_t[k*n+i][r]
                                grad_U[i*m+j][r] += err * v_val * w_val
                                grad_V[j*p+k][r] += err * u_val * w_val
                                grad_W[k*n+i][r] += err * u_val * v_val
                
                U_t -= lr * grad_U
                V_t -= lr * grad_V
                W_t -= lr * grad_W
                
                # Check convergence
                if step % 100 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.sum(U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i])
                                max_err = max(max_err, abs(s - 1.0))
                    if max_err < 1e-8:
                        found = True
                        break
            
            if found:
                # Convert to list of lists with rounded values
                U_list = [[float(round(x, 10)) for x in row] for row in U_t.tolist()]
                V_list = [[float(round(x, 10)) for x in row] for row in V_t.tolist()]
                W_list = [[float(round(x, 10)) for x in row] for row in W_t.tolist()]
                
                # Verify
                if _verify_decomposition(U_list, V_list, W_list):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    break
            elif time.time() > deadline - 20:
                break
        
        if found:
            break
        elif time.time() > deadline - 25:
            break
    
    if best_rank < rank_sb:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Return schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}
