import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # We search for rank r. Start with r=20.
    # If not found, try r=22, etc.
    # If time runs out, return schoolbook rank 30.
    
    def get_schoolbook():
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

    # Randomized search for rank 20
    # We use integer entries scaled by 2 (so values are 0, 1, 2... representing 0, 0.5, 1...)
    # Actually, let's just use integers and check if the sum is 2? No, the target is 1.
    # If we use integers, the sum must be 1.
    # Let's try small integers: -2, -1, 0, 1, 2.
    
    best_rank = n * m * p
    best_decomp = get_schoolbook()
    
    for rank in [20, 22, 24]:
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this rank
        found = False
        attempts = 0
        max_attempts = 500 # Adjust based on time
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Initialize U, V, W with random small integers
            # We can fix one matrix to simplify (gauge freedom)
            # Let's fix W to be sparse or random.
            # Actually, just randomize all.
            
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Randomly assign values
            for c in range(rank):
                # Pick random non-zero patterns
                # To ensure we don't get stuck in trivial zeros
                for i in range(n * m):
                    if random.random() < 0.3:
                        U[i, c] = random.choice([-2, -1, 1, 2])
                for j in range(m * p):
                    if random.random() < 0.3:
                        V[j, c] = random.choice([-2, -1, 1, 2])
                for k in range(p * n):
                    if random.random() < 0.3:
                        W[k, c] = random.choice([-2, -1, 1, 2])
            
            # Local search / Gradient descent
            # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
            # Error E[i,j,k] = 1 - sum_c ...
            
            # Compute current error
            # This is slow in pure Python, so we use NumPy
            # Tensor T_decomp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # We can iterate to minimize error.
            # Pick random i,j,k, and adjust U[i*m+j, c] or V or W to reduce error.
            
            # Let's do a few steps of local search
            for step in range(100):
                if time.time() > deadline - 2:
                    break
                    
                # Check if error is zero
                # Efficient check:
                # We can't easily check all 30 entries in pure Python fast enough if we do it every step.
                # But 30 is small.
                
                # Compute error
                errors = np.zeros((n, m, p), dtype=int)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            errors[i, j, k] = 1 - val
                
                if np.all(errors == 0):
                    found = True
                    best_rank = rank
                    # Convert numpy to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_decomp = {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                # Find max error entry
                idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                i, j, k = idx
                err = errors[i, j, k]
                
                if err == 0:
                    # Pick random non-zero error
                    non_zero = np.nonzero(errors)
                    if len(non_zero[0]) == 0:
                        break
                    idx_r = random.randint(0, len(non_zero[0]) - 1)
                    i = non_zero[0][idx_r]
                    j = non_zero[1][idx_r]
                    k = non_zero[2][idx_r]
                    err = errors[i, j, k]
                
                # Try to fix this error by adjusting one term
                # We can adjust U[i*m+j, c], V[j*p+k, c], or W[k*n+i, c]
                # Let's pick a random column c
                c = random.randint(0, rank - 1)
                
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                # Current contribution of column c to this entry
                contrib = U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                
                # We want new_contrib such that err - (new_contrib - contrib) is closer to 0
                # i.e., new_contrib - contrib = -err
                # new_contrib = contrib - err
                
                # We have freedom to change one of the three factors.
                # Let's try changing U.
                # U_new * V * W = contrib - err
                # If V*W is not 0, U_new = (contrib - err) / (V*W)
                
                v_w = V[idx_v, c] * W[idx_w, c]
                if v_w != 0:
                    target_u = (contrib - err) / v_w
                    # Check if target_u is close to an integer
                    if abs(target_u - round(target_u)) < 1e-9:
                        U[idx_u, c] = int(round(target_u))
                        continue
                
                # Try changing V
                u_w = U[idx_u, c] * W[idx_w, c]
                if u_w != 0:
                    target_v = (contrib - err) / u_w
                    if abs(target_v - round(target_v)) < 1e-9:
                        V[idx_v, c] = int(round(target_v))
                        continue
                        
                # Try changing W
                u_v = U[idx_u, c] * V[idx_v, c]
                if u_v != 0:
                    target_w = (contrib - err) / u_w if u_w != 0 else 0 # Typo in logic, should be u_v
                    if u_v != 0:
                        target_w = (contrib - err) / u_v
                        if abs(target_w - round(target_w)) < 1e-9:
                            W[idx_w, c] = int(round(target_w))
                            continue
                            
                # If no simple fix found, random perturbation
                if random.random() < 0.1:
                    U[idx_u, c] += random.choice([-1, 1])
                elif random.random() < 0.1:
                    V[idx_v, c] += random.choice([-1, 1])
                elif random.random() < 0.1:
                    W[idx_w, c] += random.choice([-1, 1])

        if found:
            break

    return best_decomp
