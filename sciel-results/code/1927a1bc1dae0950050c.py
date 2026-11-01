import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    n, m, p = 2, 2, 11
    
    # Target rank to try first
    target_rank = 38
    
    # Build the multiplication tensor T as a sparse representation
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We only need the non-zero entries to check correctness
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Function to compute error for given U, V, W
    def compute_error(U, V, W, rank):
        # U: (4, rank), V: (22, rank), W: (22, rank)
        error = 0.0
        # We check all 44 non-zero entries must be 1
        # And all other entries must be 0.
        # Total possible (a, b, c) tuples is 4 * 22 * 22 = 1936.
        # But we only care about the 44 ones and the rest being 0.
        # To be efficient, we can just check the 44 ones and a sample of zeros, 
        # or better, compute the full tensor contraction and check.
        
        # For speed in the loop, we will check the 44 ones strictly.
        # And we will assume if the ones are 1, the zeros might be non-zero, 
        # so we must check zeros too.
        
        # Let's use a vectorized approach for the check if possible, 
        # but for small rank, explicit loop is okay if optimized.
        
        # Actually, let's just check the 44 ones first. If they are not 1, error is high.
        # Then check a subset of zeros or all zeros.
        
        # Pre-compute indices for speed
        # U: 4 rows, V: 22 rows, W: 22 rows
        
        # Calculate sum_r U[a,r] V[b,r] W[c,r] for each (a,b,c)
        
        # This is effectively computing the tensor product.
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # But this creates a 4x22x22 array (1936 elements).
        # We can do this with numpy which is fast.
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Check non-zero entries
        err = 0.0
        for a, b, c in tensor_entries:
            val = T_calc[a, b, c]
            err += (val - 1.0) ** 2
            
        # Check zero entries (sample or all)
        # To save time, we can check all if numpy is fast enough.
        # 1936 checks is negligible.
        # Create a mask of zero entries
        # It's easier to just iterate over all and subtract the non-zero ones
        # Or just build a full target tensor T_target
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for a, b, c in tensor_entries:
            T_target[a, b, c] = 1.0
            
        diff = T_calc - T_target
        err += np.sum(diff ** 2)
        
        return err

    # Random search for rank 38
    U = np.zeros((n*m, target_rank), dtype=np.float64)
    V = np.zeros((m*p, target_rank), dtype=np.float64)
    W = np.zeros((p*n, target_rank), dtype=np.float64)
    
    # Initialize with random integers -1, 0, 1
    # Bias towards 0 to keep sparsity
    for _ in range(target_rank):
        # Randomly pick a few non-zero entries
        # For U (4 entries per column)
        for i in range(n*m):
            if random.random() < 0.3:
                U[i, _] = random.choice([-1, 1])
        # For V (22 entries per column)
        for i in range(m*p):
            if random.random() < 0.1:
                V[i, _] = random.choice([-1, 1])
        # For W (22 entries per column)
        for i in range(p*n):
            if random.random() < 0.1:
                W[i, _] = random.choice([-1, 1])
                
    best_error = compute_error(U, V, W, target_rank)
    
    # Local search / Gradient Descent step
    # Since we want integer/half-integer solutions, we can do a discrete local search.
    # Try flipping one entry in U, V, or W and see if error decreases.
    
    max_iterations = 5000
    step_size = 1.0
    
    for iter in range(max_iterations):
        if time.time() > deadline - 5:
            break
        if best_error < 1e-6:
            break
            
        # Pick a random variable to perturb
        # 1/3 chance U, 1/3 V, 1/3 W
        var_type = random.randint(0, 2)
        if var_type == 0:
            row = random.randint(0, n*m - 1)
            col = random.randint(0, target_rank - 1)
            old_val = U[row, col]
            # Try changing to -1, 0, 1
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            U[row, col] = new_val
            new_error = compute_error(U, V, W, target_rank)
            if new_error < best_error:
                best_error = new_error
            else:
                U[row, col] = old_val
        elif var_type == 1:
            row = random.randint(0, m*p - 1)
            col = random.randint(0, target_rank - 1)
            old_val = V[row, col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            V[row, col] = new_val
            new_error = compute_error(U, V, W, target_rank)
            if new_error < best_error:
                best_error = new_error
            else:
                V[row, col] = old_val
        else:
            row = random.randint(0, p*n - 1)
            col = random.randint(0, target_rank - 1)
            old_val = W[row, col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val: continue
            W[row, col] = new_val
            new_error = compute_error(U, V, W, target_rank)
            if new_error < best_error:
                best_error = new_error
            else:
                W[row, col] = old_val
                
    # If we found a solution with rank 38
    if best_error < 1e-6:
        # Convert to list of lists of ints/floats as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Clean up near-zero values
        for i in range(len(U_list)):
            for j in range(len(U_list[i])):
                if abs(U_list[i][j]) < 1e-9:
                    U_list[i][j] = 0
                elif abs(U_list[i][j] - round(U_list[i][j])) < 1e-9:
                    U_list[i][j] = int(round(U_list[i][j]))
        for i in range(len(V_list)):
            for j in range(len(V_list[i])):
                if abs(V_list[i][j]) < 1e-9:
                    V_list[i][j] = 0
                elif abs(V_list[i][j] - round(V_list[i][j])) < 1e-9:
                    V_list[i][j] = int(round(V_list[i][j]))
        for i in range(len(W_list)):
            for j in range(len(W_list[i])):
                if abs(W_list[i][j]) < 1e-9:
                    W_list[i][j] = 0
                elif abs(W_list[i][j] - round(W_list[i][j])) < 1e-9:
                    W_list[i][j] = int(round(W_list[i][j]))
                    
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Rank 39 Construction
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0       
        v_f = k1       
        v_g = 11 + k0  
        v_h = 11 + k1  
        
        w_c00 = 2 * k0     
        w_c10 = 2 * k0 + 1 
        w_c01 = 2 * k1     
        w_c11 = 2 * k1 + 1 
        
        # m1
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

    # Last column k=10: schoolbook
    k2 = 10
    v_e2 = k2      
    v_g2 = 11 + k2  
    w_c02 = 2 * k2      
    w_c12 = 2 * k2 + 1  

    U[0][r] += 1
    V[v_e2][r] += 1
    W[w_c02][r] += 1
    r += 1

    U[1][r] += 1
    V[v_g2][r] += 1
    W[w_c02][r] += 1
    r += 1

    U[2][r] += 1
    V[v_e2][r] += 1
    W[w_c12][r] += 1
    r += 1

    U[3][r] += 1
    V[v_g2][r] += 1
    W[w_c12][r] += 1
    r += 1

    return {"U": U, "V": V, "W": W}
