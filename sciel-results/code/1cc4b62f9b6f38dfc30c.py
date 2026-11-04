import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin

    # Schoolbook decomposition as fallback
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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Exploit the commutative symmetry of matrix multiplication.
    # The tensor for (n,m,p) = (3,3,8) can be related to (p,m,n) = (8,3,3).
    # For (8,3,3), we can use a known efficient decomposition.
    # The rank-21 algorithm for 8x3x3 is derived from Strassen's 7 for 2x2x2
    # and extensions. Let's try to construct it.
    
    # Actually, let's try a different approach: use the fact that 8 = 2^3
    # and use tensor product decompositions.
    
    # For a 3x3 matrix times 3x8 matrix, we can view the 3x8 as three 3x3 blocks
    # and one 3x2 block.
    # A 3x3 x 3x3 multiplication has rank at most 21 (using Laderman's algorithm
    # or similar). But the standard is 27.
    
    # Let's try to find a decomposition by exploiting structure.
    # The key insight: if we can find a decomposition where the W matrix has
    # a special structure, we can reduce the rank.
    
    # Alternative: Use the fact that the tensor is the Kronecker product of
    # smaller tensors. 3x3 x 3x8 = (3x3 x 3x3) ⊗ (1x1 x 1x8) doesn't help.
    
    # Let's try a more systematic search with better initialization.
    # Use the structure of the problem: the tensor T_{i,j,k} = delta_{i,k} for
    # the multiplication tensor.
    
    # For matrix multiplication, T_{i,j,k} = 1 if i=k (in the output), 0 otherwise.
    # Wait, that's not right. The multiplication tensor is T_{i,j,k} = 1 if the
    # (i,k)-th entry of the product involves A_{i,j} * B_{j,k}.
    
    # The tensor is: T_{(i,j),(j,k),(k,i)} = 1 for all valid indices, 0 otherwise.
    # In our indexing: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1
    # for all (i,j,k) in range, and the tensor is fully defined.
    
    # Let's try a rank-56 decomposition using a structured approach.
    # The idea: use the fact that 8 = 2+2+2+2 and decompose into four 3x3x3
    # problems, but share factors where possible.
    
    # Actually, let's try to use a known result: the rank of 3x3 x 3x8 is at
    # most 56 (by a result of Bini et al. or similar).
    
    # Let's try a random search with better constraints.
    # Initialize with small integer/half-integer values and use local search.
    
    # First, let's try to find a decomposition with rank 56.
    # We'll use a simulated annealing approach with the constraint that entries
    # are half-integers.
    
    # For efficiency, let's use numpy for the verification.
    
    def verify_fast(U, V, W, rank):
        """Fast verification using numpy."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Compute the tensor contraction
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape for efficient computation
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a 4-index contraction: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # We can compute this as:
        # For each c, compute the outer product U_c ⊗ V_c ⊗ W_c and sum.
        # But that's expensive. Instead, for each (i,j), compute:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # Let's do it in a loop over i, j, and use numpy for k.
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # = sum_k (sum_c U[i,j,c] * V[j,k,c] * W[k,i,c])
                # For fixed i,j: T[i,j,:] = sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c])
                
                # V[j,:,c] has shape (p,) for each c
                # W[:,i,c] has shape (p,) for each c
                # So V[j,:,c] * W[:,i,c] has shape (p,)
                # Then sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c])
                
                # Let's compute: for each c, compute V[j,:,c] * W[:,i,c],
                # then multiply by U[i,j,c] and sum over c.
                
                # V_reshaped[j, :, :] has shape (p, rank)
                # W_reshaped[:, i, :] has shape (p, rank)
                # Element-wise product: (p, rank)
                # Multiply by U[i,j,:] (rank,) and sum over rank: (p,)
                
                prod = V_reshaped[j, :, :] * W_reshaped[:, i, :]  # (p, rank)
                T_ij = np.sum(prod * U_reshaped[i, j, :], axis=1)  # (p,)
                
                # Check if T_ij is all 1s
                if not np.allclose(T_ij, 1.0, atol=1e-9):
                    return False
        
        return True
    
    # Try ranks from 71 down to 56
    for rank in range(71, 55, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with integer/half-integer values
            # Use small values to increase chance of exact match
            U = np.random.randint(-2, 3, (n * m, rank)) * 0.5
            V = np.random.randint(-2, 3, (m * p, rank)) * 0.5
            W = np.random.randint(-2, 3, (p * n, rank)) * 0.5
            
            # Check if this decomposes the tensor exactly
            if verify_fast(U, V, W, rank):
                # Convert to list of lists with integer/half-integer values
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Round to nearest half-integer
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                # Verify after rounding
                if verify_fast(U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found a valid decomposition for this rank
    
    if time.time() > deadline:
        return best_result
    
    return best_result
