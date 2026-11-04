import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a low-rank decomposition
    # Start with the seed (rank 27) and try to reduce
    
    # The known minimal rank for 3x3 matrix multiplication is 23.
    # We'll attempt to construct a valid rank-23 decomposition.
    
    # Winograd's rank-23 decomposition for 3x3 matrix multiplication
    # uses the following approach: embed in 4x4 and use specific structure.
    
    # Let's try a direct construction. The 3x3 matmul tensor has entries
    # T[i,j,k] = delta_{valid} where the tensor is indexed by (i,j) for A,
    # (j,k) for B, and (k,i) for C.
    
    # We need U (9 x r), V (9 x r), W (9 x r) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if valid, 0 otherwise
    
    # For the standard 3x3 matmul, all 27 combinations (i,j,k) with i,j,k in {0,1,2}
    # are valid, so T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    
    # Wait - that means the tensor is just the all-ones 9x9x9 tensor?
    # No! The tensor for matrix multiplication is:
    # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    # But the indexing is: A has indices (i,j), B has indices (j,k), C has indices (k,i)
    # So T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # Which means T[(i,j),(j,k),(k,i)] = 1
    
    # In the flattened 9x9x9 tensor with indices:
    # A index: i*m+j = i*3+j
    # B index: j*p+k = j*3+k  
    # C index: k*n+i = k*3+i
    
    # So T[(i*3+j), (j*3+k), (k*3+i)] = 1 for all i,j,k in {0,1,2}
    # And T is 0 elsewhere.
    
    # This is NOT the all-ones tensor. It's a specific pattern.
    
    # The seed decomposition uses rank 27 with U,V,W being one-hot vectors.
    # To get rank 23, we need to find 23 rank-1 tensors that sum to T.
    
    # Let's use a numerical approach: start with random initialization and optimize.
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U,V,W form a valid decomposition of the 3x3 matmul tensor."""
        n, m, p = 3, 3, 3
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For efficiency, use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Build the target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Check if they match
        if not np.allclose(T_recon, T_target, atol=tol):
            return False
        return True
    
    def to_int_matrices(U, V, W):
        """Convert numpy arrays to lists of lists, rounding to nearest integer if close."""
        U_list = [[int(round(val)) if abs(val - round(val)) < 1e-6 else val for val in row] for row in U]
        V_list = [[int(round(val)) if abs(val - round(val)) < 1e-6 else val for val in row] for row in V]
        W_list = [[int(round(val)) if abs(val - round(val)) < 1e-6 else val for val in row] for row in W]
        return U_list, V_list, W_list
    
    # Try to find rank-23 decomposition using optimization
    # We'll use a simple iterative approach
    
    best_result = None
    best_rank = 27  # seed rank
    
    # First, ensure we have a valid seed
    seed_result = construct_seed()
    best_result = seed_result
    best_rank = 27
    
    # Try to find lower rank decompositions
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(9, target_rank) * 0.1
            V = np.random.randn(9, target_rank) * 0.1
            W = np.random.randn(9, target_rank) * 0.1
            
            # Optimize using gradient descent
            # Objective: minimize ||T_recon - T_target||^2
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                # Build target once
                if step == 0:
                    T_target = np.zeros((9, 9, 9))
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                T_target[i*3+j, j*3+k, k*3+i] = 1.0
                
                diff = T_recon - T_target
                loss = np.sum(diff ** 2)
                
                if loss < 1e-8:
                    # Check exactness
                    if verify_decomposition(U, V, W, tol=1e-9):
                        U_list, V_list, W_list = to_int_matrices(U, V, W)
                        result = {"U": U_list, "V": V_list, "W": W_list}
                        if best_result is None or target_rank < best_rank:
                            best_result = result
                            best_rank = target_rank
                        break
                    elif np.allclose(T_recon, T_target, atol=1e-6):
                        # Close enough, try to snap to integers
                        U_snapped = np.round(U)
                        V_snapped = np.round(V)
                        W_snapped = np.round(W)
                        if verify_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-9):
                            U_list, V_list, W_list = to_int_matrices(U_snapped, V_snapped, W_snapped)
                            result = {"U": U_list, "V": V_list, "W": W_list}
                            if best_result is None or target_rank < best_rank:
                                best_result = result
                                best_rank = target_rank
                        break
                
                # Gradients
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                if step % 500 == 499 and step > 0:
                    lr *= 0.5
            
            if time.time() > deadline - 5:
                break
    
    if best_result is not None:
        return best_result
    else:
        return construct_seed()

def construct_seed():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
