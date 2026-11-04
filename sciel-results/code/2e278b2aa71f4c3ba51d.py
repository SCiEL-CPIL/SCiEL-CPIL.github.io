import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Build the target tensor: T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))
    
    # Precompute all triples and invalid list
    all_triples = set()
    for a in range(num_A):
        for b in range(num_B):
            for c in range(num_C):
                all_triples.add((a, b, c))
    invalid_triples = all_triples - valid_triples
    invalid_list = list(invalid_triples)
    
    def verify_decomp(U, V, W):
        """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid triples
        for (a, b, c) in valid_triples:
            val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check ALL invalid triples
        for (a, b, c) in invalid_list:
            val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
            if abs(val) > 1e-9:
                return False
        
        return True
    
    # Strategy 1: Structured Kronecker decomposition exploiting 6 = 2*3
    # The 2x3x6 multiplication can be decomposed using the structure of the dimensions.
    # Think of B as a 3x(2*3) matrix. We can use a block structure.
    
    # Schoolbook: rank = 36
    best_rank = n * m * p
    U = np.zeros((num_A, best_rank))
    V = np.zeros((num_B, best_rank))
    W = np.zeros((num_C, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 2: Try structured decompositions with lower rank
    # The key idea: exploit that p=6=2*3 to use a tensor product structure
    
    if time.time() < deadline:
        # Try a structured approach: split the k-index into two parts
        # k = k1 * 3 + k2 where k1 in {0,1}, k2 in {0,1,2}
        # This gives a 2x3x(2*3) tensor that might be decomposable as a sum of
        # smaller tensor products
        
        # Attempt: rank 30 decomposition using block structure
        # Split into 3 blocks of 2x3x2 (each block handles 2 values of k)
        # Each 2x3x2 block can potentially be done in rank 8 (schoolbook) or less
        
        # Actually, let's try a more principled approach:
        # The multiplication tensor for (n,m,p) has rank at most n*m*p.
        # For (2,3,6), we can try to find a decomposition of rank < 36.
        
        # Use a parameterized family based on the structure
        # Let's try rank 33: save 3 terms by combining some operations
        
        # Direct construction attempt for rank 33:
        # Use the fact that for each (i,j), we need to compute sum over k of A[i,j]*B[j,k]
        # We can share some computation across different (i,j) pairs
        
        # Try multiple candidate ranks with a smarter search
        for target_rank in [33, 32, 31, 30, 29, 28, 27, 26, 25, 24]:
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(5):
                if time.time() > deadline:
                    break
                
                seed = 42 + target_rank * 1000 + restart * 7
                np.random.seed(seed)
                
                # Initialize with schoolbook and then perturb
                U_curr = np.zeros((num_A, target_rank))
                V_curr = np.zeros((num_B, target_rank))
                W_curr = np.zeros((num_C, target_rank))
                
                # Start with first target_rank schoolbook terms
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= target_rank:
                                break
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            U_curr[a, col] = 1
                            V_curr[b, col] = 1
                            W_curr[c, col] = 1
                            col += 1
                            if col >= target_rank:
                                break
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                
                # Add small random perturbation to explore
                U_curr += np.random.randn(num_A, target_rank) * 0.01
                V_curr += np.random.randn(num_B, target_rank) * 0.01
                W_curr += np.random.randn(num_C, target_rank) * 0.01
                
                # Use a more efficient optimization: alternating least squares style
                # but with the full constraint set
                
                lr = 0.02
                best_res = float('inf')
                best_U_state = U_curr.copy()
                best_V_state = V_curr.copy()
                best_W_state = W_curr.copy()
                
                # Use a subset of invalid triples for speed
                if len(invalid_list) > 200:
                    np.random.seed(seed + 1000)
                    invalid_indices = np.random.choice(len(invalid_list), 200, replace=False)
                    invalid_subset = [invalid_list[i] for i in invalid_indices]
                else:
                    invalid_subset = invalid_list
                
                for step in range(500):
                    if time.time() > deadline:
                        break
                    
                    grad_U = np.zeros_like(U_curr)
                    grad_V = np.zeros_like(V_curr)
                    grad_W = np.zeros_like(W_curr)
                    residual_sum = 0
                    
                    # Valid triples: must equal 1
                    for (a, b, c) in valid_triples:
                        val = np.dot(U_curr[a, :] * V_curr[b, :], W_curr[c, :])
                        err = val - 1.0
                        residual_sum += err * err
                        grad_U[a, :] += err * V_curr[b, :] * W_curr[c, :]
                        grad_V[b, :] += err * U_curr[a, :] * W_curr[c, :]
                        grad_W[c, :] += err * U_curr[a, :] * V_curr[b, :]
                    
                    # Invalid triples: must equal 0
                    for (a, b, c) in invalid_subset:
                        val = np.dot(U_curr[a, :] * V_curr[b, :], W_curr[c, :])
                        err = val
                        residual_sum += err * err
                        grad_U[a, :] += err * V_curr[b, :] * W_curr[c, :]
                        grad_V[b, :] += err * U_curr[a, :] * W_curr[c, :]
                        grad_W[c, :] += err * U_curr[a, :] * V_curr[b, :]
                    
                    # Update
                    U_curr -= lr * grad_U
                    V_curr -= lr * grad_V
                    W_curr -= lr * grad_W
                    
                    lr *= 0.997
                    
                    if residual_sum < best_res:
                        best_res = residual_sum
                        best_U_state = U_curr.copy()
                        best_V_state = V_curr.copy()
                        best_W_state = W_curr.copy()
                    
                    if residual_sum < 1e-12:
                        found = True
                        break
                
                # Full verification
                if verify_decomp(best_U_state, best_V_state, best_W_state):
                    best_rank = target_rank
                    best_U = best_U_state.tolist()
                    best_V = best_V_state.tolist()
                    best_W = best_W_state.tolist()
                    found = True
                    break
            
            if found:
                break
    
    # Final safety check
    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook
        U = np.zeros((num_A, best_rank))
        V = np.zeros((num_B, best_rank))
        W = np.zeros((num_C, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
