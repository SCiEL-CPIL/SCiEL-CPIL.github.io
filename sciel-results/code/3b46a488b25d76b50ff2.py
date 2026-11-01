import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140  # 140s out of 150s, leave 10s safety margin
    
    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return U, V, W
    
    # Build the target tensor
    # T[i, j, k] = 1 for the multiplication tensor
    # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this term contributes to C[i,k] = sum_j A[i,j]B[j,k]
    # Actually, the tensor T is defined such that T[i,j,k] = 1 for all valid (i,j,k)
    # and the reconstruction is: C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
    # Wait, I need to be more careful.
    
    # The standard tensor decomposition for matrix multiplication:
    # C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_j sum_r (U[i*m+j, r] A[i,j]) (V[j*p+k, r] B[j,k]) W[k*n+i, r]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    # No wait, that's not right either.
    
    # Let me reconsider. The correct formulation:
    # We want: for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
    # Using the decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # This should equal sum_j A[i,j] B[j,k]
    
    # For the schoolbook, each r corresponds to a unique (i0, j0, k0):
    # U[i*m+j, r] = 1 iff i=i0 and j=j0
    # V[j*p+k, r] = 1 iff j=j0 and k=k0  
    # W[k*n+i, r] = 1 iff k=k0 and i=i0
    # Then the term for r is: A[i0,j0] * B[j0,k0] * W[k*n+i, r]
    # This is non-zero only when i=i0 and k=k0, giving A[i,j0]*B[j0,k]*1
    # Summing over all r (i.e., all j0) gives sum_j0 A[i,j0]*B[j0,k] = C[i,k]
    
    # So the condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{contributing}
    # Actually, it's not simply 1. The condition is that the bilinear form reconstructs correctly.
    
    # Let me define the error function properly.
    # Given U, V, W, the reconstructed C[i,k] is:
    # C_recon[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # We want this to equal sum_j A[i,j]*B[j,k] for ALL A and B.
    # This means: for all i,k: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) = sum_j A[i,j] B[j,k]
    
    # This is a polynomial identity in A and B. For it to hold for all A, B, we need:
    # The coefficient of A[i',j'] * B[j'',k'] in the left side must equal the coefficient on the right side.
    # Right side: coefficient of A[i',j'] * B[j'',k'] is 1 if i'=i, j'=j'', k'=k, else 0.
    # Left side: sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] (with the sums absorbed)
    # So we need: for all i,i',j',j'',k,k': 
    #   sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j''*p+k', r] = 1 if i'=i, j'=j'', k'=k else 0
    
    # This is a system of equations. Let me set up the target.
    
    # The number of constraints: n * n * m * m * p * p = 2*2*4*4*8*8 = 4096 constraints
    # But many are redundant. Actually, the independent constraints are:
    # For each (i, j, k): the "diagonal" constraint that must equal 1
    # For all other (i, j, k) combinations: must equal 0
    
    # Actually, let me just use a random A and B to check correctness, and use that as the objective.
    
    def compute_error(U, V, W, A, B):
        """Compute the error between reconstructed and true product."""
        # C_true[i,k] = sum_j A[i,j] B[j,k]
        C_true = A @ B
        # C_recon[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
        # Precompute: X[i, r] = sum_j U[i*m+j, r] A[i,j]
        # Y[k, r] = sum_j V[j*p+k, r] B[j,k]
        # Then C_recon[i,k] = sum_r X[i, r] * Y[k, r] * W[k*n+i, r]
        
        # X[i, r] = sum_j U[i*m+j, r] * A[i,j]
        # Reshape U to (n, m, rank)
        U_reshaped = U.reshape(n, m, -1)
        # X[i, r] = sum_j U_reshaped[i, j, r] * A[i, j]
        X = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        # Y[k, r] = sum_j V[j*p+k, r] * B[j,k]
        # Reshape V to (m, p, rank)
        V_reshaped = V.reshape(m, p, -1)
        # Y[k, r] = sum_j V_reshaped[j, k, r] * B[j, k]
        Y = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        # C_recon[i,k] = sum_r X[i, r] * Y[k, r] * W[k*n+i, r]
        # Reshape W to (p, n, rank)
        W_reshaped = W.reshape(p, n, -1)
        # C_recon[i,k] = sum_r X[i, r] * Y[k, r] * W_reshaped[k, i, r]
        C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
        
        return np.sum((C_recon - C_true) ** 2)
    
    def verify_decomposition(U, V, W, num_tests=10, seed=42):
        """Verify the decomposition with random matrices."""
        rng = np.random.RandomState(seed)
        max_err = 0
        for _ in range(num_tests):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            err = compute_error(U, V, W, A, B)
            max_err = max(max_err, err)
        return max_err
    
    # Try to find lower-rank decompositions
    best_rank = n * m * p
    best_U, best_V, best_W = get_schoolbook()
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Actually, the lower bound is at least max(n*m, m*p, p*n) / something, but let's be conservative
    # Known: rank >= max(n*m, m*p, p*n) is not necessarily true, but rank >= n*m*p / (n+m+p-2) or similar
    # Let's try from rank 32 up to 64
    
    for rank in range(32, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(rank * 1000 + trial)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Check initial error
            A_test = np.random.RandomState(0).randn(n, m)
            B_test = np.random.RandomState(1).randn(m, p)
            err = compute_error(U, V, W, A_test, B_test)
            
            # Gradient descent
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute gradients numerically or analytically
                # For simplicity, use finite differences for a few dimensions
                # Actually, let's use analytic gradients
                
                # C_recon[i,k] = sum_r X[i,r] Y[k,r] W[k*n+i, r]
                # where X[i,r] = sum_j U[i*m+j, r] A[i,j]
                #       Y[k,r] = sum_j V[j*p+k, r] B[j,k]
                
                # d(C_recon[i,k])/d(U[a*m+b, r]) = A[a,b] * Y[k,r] * W[k*n+a, r] if a=i, else 0
                # Actually: d(C_recon[i,k])/d(U[i*m+j, r]) = A[i,j] * Y[k,r] * W[k*n+i, r]
                
                # Let me compute gradients analytically
                U_reshaped = U.reshape(n, m, -1)
                V_reshaped = V.reshape(m, p, -1)
                W_reshaped = W.reshape(p, n, -1)
                
                X = np.einsum('ijr,ij->ir', U_reshaped, A_test)
                Y = np.einsum('jkr,jk->kr', V_reshaped, B_test)
                C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
                C_true = A_test @ B_test
                
                # Gradient of error = 2 * sum of (C_recon - C_true) * dC_recon/d(param)
                diff = C_recon - C_true  # shape (n, p)
                
                # dU[i*m+j, r] = 2 * sum_k diff[i,k] * A[i,j] * Y[k,r] * W[k*n+i, r]
                # = 2 * A[i,j] * sum_k diff[i,k] * Y[k,r] * W[k*n+i, r]
                # Let Z[i,r] = sum_k diff[i,k] * Y[k,r] * W[k*n+i, r]
                Z = np.einsum('ik,kr,ki->ir', diff, Y, W_reshaped)
                dU = 2 * np.einsum('ij,ir->ijr', A_test, Z)
                dU = dU.reshape(n * m, rank)
                
                # dV[j*p+k, r] = 2 * sum_i diff[i,k] * X[i,r] * B[j,k] * W[k*n+i, r]
                # = 2 * B[j,k] * sum_i diff[i,k] * X[i,r] * W[k*n+i, r]
                # Let Z2[k,r] = sum_i diff[i,k] * X[i,r] * W[k*n+i, r]
                Z2 = np.einsum('ik,ir,ki->kr', diff, X, W_reshaped)
                dV = 2 * np.einsum('jk,kr->jkr', B_test, Z2)
                dV = dV.reshape(m * p, rank)
                
                # dW[k*n+i, r] = 2 * sum_i diff[i,k] * X[i,r] * Y[k,r]
                # = 2 * sum_i diff[i,k] * X[i,r] * Y[k,r]
                dW_reshaped = 2 * np.einsum('ik,ir,kr->ki', diff, X, Y)
                dW = dW_reshaped.reshape(p * n, rank)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                err = compute_error(U, V, W, A_test, B_test)
                
                if err < 1e-10:
                    break
            
            # Verify with multiple random matrices
            if err < 1e-8:
                max_verify_err = verify_decomposition(U, V, W, num_tests=20)
                if max_verify_err < 1e-6:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Found a valid decomposition at this rank
    
    # Return the best found
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
