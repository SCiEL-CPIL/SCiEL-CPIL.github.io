"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Verify a decomposition
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition
    def schoolbook_decomp():
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
        return U, V, W, rank

    # Strategy: Model 3x3x6 as direct sum of two 3x3x3 tensors
    # For 3x3x3, we use a known rank-23 decomposition (Laderman's algorithm)
    # Total rank would be 2*23 = 46
    
    # Laderman's algorithm for 3x3x3 matrix multiplication
    # This gives rank 23 for multiplying two 3x3 matrices
    
    # The 3x3x3 tensor T[i,j,k] = delta_{i,j,k} where C[i,k] = sum_j A[i,j]*B[j,k]
    # In our convention: U[i*m+j], V[j*p+k], W[k*n+i]
    # For 3x3x3: n=m=p=3, so U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Laderman's 23-term algorithm for 3x3x3:
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The 23 products (Laderman's algorithm):
    # We need to express each C[i,k] as sum of products from the 23 terms
    
    # Here's a verified rank-23 decomposition for 3x3x3:
    # Using the standard Laderman construction
    
    def laderman_3x3x3():
        """Return U, V, W matrices (each 9x23) for 3x3x3 multiplication with rank 23"""
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Laderman's algorithm coefficients
        # The 23 terms are defined by linear combinations of A and B entries
        # and their products contribute to C entries
        
        # Let me use a known explicit construction.
        # Reference: Laderman, S. (1976). "On the number of multiplications needed 
        # to multiply 3x3 matrices."
        
        # I'll construct this carefully using the known rank-23 formula.
        # The algorithm uses 23 scalar multiplications.
        
        # Define the 23 products as:
        # p_t = (sum_i alpha_t[i]*A[i]) * (sum_k beta_t[k]*B[k])
        # where A[i] and B[k] are row/column vectors, and C = sum_t p_t * gamma_t
        
        # Actually, let me use a different, more direct approach.
        # I'll use the fact that we can verify any candidate decomposition.
        
        # For now, let me try a structured approach:
        # Use Kronecker product structure or known low-rank decompositions.
        
        # Alternative: Use the fact that 3x3x3 has known rank 23.
        # I'll hardcode a verified rank-23 decomposition.
        
        # Here's a known rank-23 decomposition for 3x3x3:
        # Based on Laderman's algorithm
        
        # The 23 terms (using 0-indexed):
        # Let me define them explicitly.
        
        # Actually, let me use a simpler approach: 
        # I'll construct a valid decomposition and verify it.
        # If it works, great. If not, fall back to schoolbook.
        
        # For 3x3x3, one known rank-23 decomposition:
        # I'll use the explicit coefficients from literature.
        
        # Let me try a different strategy: 
        # Use random initialization with gradient descent for 3x3x3 first,
        # then extend to 3x3x6.
        
        return None, None, None, rank  # Placeholder
    
    # Better approach: Directly search for 3x3x6 decomposition
    # Start with rank 46 (2 * 23) and try to find a valid decomposition
    
    # Initialize with block structure: two independent 3x3x3 decompositions
    # For the first 3 columns of B (k=0,1,2) and second 3 columns (k=3,4,5)
    
    def init_block_decomp(target_rank):
        """Initialize with block structure assuming two 3x3x3 blocks"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Assume rank = 2 * r3 where r3 is rank for 3x3x3
        # First block: columns k=0,1,2 use ranks 0..r3-1
        # Second block: columns k=3,4,5 use ranks r3..2*r3-1
        
        if target_rank % 2 == 0:
            r3 = target_rank // 2
            # Initialize each block with random values
            for block in range(2):
                offset = block * r3
                U[:, offset:offset+r3] = np.random.randn(n*m, r3) * 0.5
                V[:, offset:offset+r3] = np.random.randn(m*p, r3) * 0.5
                W[:, offset:offset+r3] = np.random.randn(p*n, r3) * 0.5
                
                # Zero out cross-block contributions
                # For block 0 (k=0,1,2): V rows for k=3,4,5 should be 0
                # For block 1 (k=3,4,5): V rows for k=0,1,2 should be 0
                if block == 0:
                    V[9:18, offset:offset+r3] = 0  # k=3,4,5 rows
                else:
                    V[0:9, offset:offset+r3] = 0  # k=0,1,2 rows
        else:
            # Odd rank: just random init
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
        
        return U, V, W
    
    # Compute the tensor values T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # using vectorized operations for speed
    def compute_tensor(U, V, W, rank):
        """Compute all 54 tensor entries using vectorized matmul"""
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # For each (i,j,k), T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can compute this as: for each r, outer product U[:, :, r] @ V[:, :, r] @ W[:, :, r]
        # But more efficiently:
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's use einsum: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Using np.einsum
        tensor = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return tensor
    
    # Compute gradients using vectorized operations
    def compute_gradients(U, V, W, rank, error):
        """Compute gradients dU, dV, dW given error tensor"""
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        error_reshaped = error.reshape(n, m, p)
        
        # dU[i*m+j, r] = sum_{i,j,k} error[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        # In reshaped form: dU[i,j,r] = sum_k error[i,j,k] * V[j,k,r] * W[k,i,r]
        dU_reshaped = np.einsum('ijk,jkr,kir->ijr', error_reshaped, V_reshaped, W_reshaped)
        
        # dV[j*p+k, r] = sum_{i,j,k} error[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
        # In reshaped form: dV[j,k,r] = sum_i error[i,j,k] * U[i,j,r] * W[k,i,r]
        dV_reshaped = np.einsum('ijk,ijr,kir->jkr', error_reshaped, U_reshaped, W_reshaped)
        
        # dW[k*n+i, r] = sum_{i,j,k} error[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
        # In reshaped form: dW[k,i,r] = sum_j error[i,j,k] * U[i,j,r] * V[j,k,r]
        dW_reshaped = np.einsum('ijk,ijr,jkr->kir', error_reshaped, U_reshaped, V_reshaped)
        
        dU = dU_reshaped.reshape(n*m, rank)
        dV = dV_reshaped.reshape(m*p, rank)
        dW = dW_reshaped.reshape(p*n, rank)
        
        return dU, dV, dW
    
    # Try ranks from 46 down to 40
    candidate_ranks = [46, 45, 44, 43, 42, 41, 40]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        for init_idx in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(100 + target_rank * 10 + init_idx)
            U, V, W = init_block_decomp(target_rank)
            
            # Gradient descent to refine with vectorized operations
            eta = 0.05
            best_local_error = np.inf
            for iteration in range(300):
                if time.time() > deadline:
                    break
                    
                # Compute tensor and error using vectorized operations
                tensor = compute_tensor(U, V, W, target_rank)
                error = 1.0 - tensor
                current_error = np.max(np.abs(error))
                
                if current_error < 1e-9:
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
                
                if current_error < best_local_error:
                    best_local_error = current_error
                    best_local_U = U.copy()
                    best_local_V = V.copy()
                    best_local_W = W.copy()
                
                # Compute gradients using vectorized operations
                dU, dV, dW = compute_gradients(U, V, W, target_rank, error)
                
                U -= eta * dU
                V -= eta * dV
                W -= eta * dW
                
                if iteration % 30 == 0:
                    eta *= 0.9
                    if eta < 0.001:
                        eta = 0.001
            
            # Check if we found a solution
            if best_rank <= target_rank:
                break
            
            # If not converged, check if local best is close
            if best_local_error < 1e-7:
                # Try to polish with smaller steps
                U, V, W = best_local_U, best_local_V, best_local_W
                eta = 0.005
                for iteration in range(100):
                    if time.time() > deadline:
                        break
                    tensor = compute_tensor(U, V, W, target_rank)
                    error = 1.0 - tensor
                    current_error = np.max(np.abs(error))
                    
                    if current_error < 1e-9:
                        best_rank = target_rank
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                        break
                    
                    dU, dV, dW = compute_gradients(U, V, W, target_rank, error)
                    U -= eta * dU
                    V -= eta * dV
                    W -= eta * dW
    
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
