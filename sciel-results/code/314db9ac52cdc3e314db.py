import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    S = [] # Non-zero indices
    Z = [] # Zero indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
                S.append((a_idx, b_idx, c_idx))
    
    # Generate all zero indices
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in S:
                    Z.append((a, b, c))
    
    # Map index to boolean for fast lookup
    S_set = set(S)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    vals = [-1, 0, 1]
    # vals = [-2, -1, 0, 1, 2]
    
    def check_decomposition(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_rec)
        return err < 1e-9

    def local_search(rank, time_limit, init_vals=None):
        end_time = min(time.time() + time_limit, deadline)
        
        # Initialize U, V, W
        if init_vals:
            U, V, W = init_vals
        else:
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                for r_idx in random.sample(range(n*m), random.randint(1, 3)):
                    U[r_idx, r] = random.choice(vals)
                for r_idx in random.sample(range(m*p), random.randint(1, 3)):
                    V[r_idx, r] = random.choice(vals)
                for r_idx in random.sample(range(p*n), random.randint(1, 3)):
                    W[r_idx, r] = random.choice(vals)
            
        # Compute initial Sums
        Sums = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Compute initial errors
        E_S = 0.0
        E_Z = 0.0
        for idx in S:
            E_S += (Sums[idx] - 1.0)**2
        for idx in Z:
            E_Z += Sums[idx]**2
            
        best_E = E_S + E_Z
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Local search
        steps = 0
        while time.time() < end_time:
            steps += 1
            if steps % 1000 == 0 and time.time() > end_time:
                break
                
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                r, c = random.randint(0, n*m-1), random.randint(0, rank-1)
                old_val = U[r, c]
                new_val = random.choice(vals)
                if old_val == new_val:
                    continue
                delta = new_val - old_val
                # Update Sums
                # Sums[r, b, c] += delta * V[b, c] * W[c, c]
                # Wait, V is (m*p, rank), W is (p*n, rank)
                # For fixed r (row of U) and c (col of U):
                # U[r, c] changes by delta.
                # This affects Sums[r, b, c'] for all b, c'.
                # Sums[r, b, c'] = sum_k U[r, k] V[b, k] W[c', k]
                # Change in Sums[r, b, c'] = delta * V[b, c] * W[c', c]
                
                # We need to update E_S and E_Z
                # Iterate over b and c'
                for b in range(m*p):
                    v_val = V[b, c]
                    if v_val == 0: continue
                    for c_prime in range(p*n):
                        w_val = W[c_prime, c]
                        if w_val == 0: continue
                        idx = (r, b, c_prime)
                        old_sum = Sums[r, b, c_prime]
                        new_sum = old_sum + delta * v_val * w_val
                        Sums[r, b, c_prime] = new_sum
                        
                        if idx in S_set:
                            E_S += (new_sum - 1.0)**2 - (old_sum - 1.0)**2
                        else:
                            E_Z += new_sum**2 - old_sum**2
                            
                U[r, c] = new_val
                curr_E = E_S + E_Z
                if curr_E < best_E:
                    best_E = curr_E
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_E < 1e-9:
                        return U, V, W
                else:
                    # Revert
                    U[r, c] = old_val
                    # Revert Sums and Errors
                    for b in range(m*p):
                        v_val = V[b, c]
                        if v_val == 0: continue
                        for c_prime in range(p*n):
                            w_val = W[c_prime, c]
                            if w_val == 0: continue
                            idx = (r, b, c_prime)
                            new_sum = Sums[r, b, c_prime]
                            old_sum = new_sum - delta * v_val * w_val
                            Sums[r, b, c_prime] = old_sum
                            if idx in S_set:
                                E_S += (old_sum - 1.0)**2 - (new_sum - 1.0)**2
                            else:
                                E_Z += old_sum**2 - new_sum**2
                                
            elif mat_idx == 1:
                r, c = random.randint(0, m*p-1), random.randint(0, rank-1)
                old_val = V[r, c]
                new_val = random.choice(vals)
                if old_val == new_val:
                    continue
                delta = new_val - old_val
                for a in range(n*m):
                    u_val = U[a, c]
                    if u_val == 0: continue
                    for c_prime in range(p*n):
                        w_val = W[c_prime, c]
                        if w_val == 0: continue
                        idx = (a, r, c_prime)
                        old_sum = Sums[a, r, c_prime]
                        new_sum = old_sum + delta * u_val * w_val
                        Sums[a, r, c_prime] = new_sum
                        if idx in S_set:
                            E_S += (new_sum - 1.0)**2 - (old_sum - 1.0)**2
                        else:
                            E_Z += new_sum**2 - old_sum**2
                V[r, c] = new_val
                curr_E = E_S + E_Z
                if curr_E < best_E:
                    best_E = curr_E
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_E < 1e-9:
                        return U, V, W
                else:
                    V[r, c] = old_val
                    for a in range(n*m):
                        u_val = U[a, c]
                        if u_val == 0: continue
                        for c_prime in range(p*n):
                            w_val = W[c_prime, c]
                            if w_val == 0: continue
                            idx = (a, r, c_prime)
                            new_sum = Sums[a, r, c_prime]
                            old_sum = new_sum - delta * u_val * w_val
                            Sums[a, r, c_prime] = old_sum
                            if idx in S_set:
                                E_S += (old_sum - 1.0)**2 - (new_sum - 1.0)**2
                            else:
                                E_Z += old_sum**2 - new_sum**2
            else:
                r, c = random.randint(0, p*n-1), random.randint(0, rank-1)
                old_val = W[r, c]
                new_val = random.choice(vals)
                if old_val == new_val:
                    continue
                delta = new_val - old_val
                for a in range(n*m):
                    u_val = U[a, c]
                    if u_val == 0: continue
                    for b in range(m*p):
                        v_val = V[b, c]
                        if v_val == 0: continue
                        idx = (a, b, r)
                        old_sum = Sums[a, b, r]
                        new_sum = old_sum + delta * u_val * v_val
                        Sums[a, b, r] = new_sum
                        if idx in S_set:
                            E_S += (new_sum - 1.0)**2 - (old_sum - 1.0)**2
                        else:
                            E_Z += new_sum**2 - old_sum**2
                W[r, c] = new_val
                curr_E = E_S + E_Z
                if curr_E < best_E:
                    best_E = curr_E
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_E < 1e-9:
                        return U, V, W
                else:
                    W[r, c] = old_val
                    for a in range(n*m):
                        u_val = U[a, c]
                        if u_val == 0: continue
                        for b in range(m*p):
                            v_val = V[b, c]
                            if v_val == 0: continue
                            idx = (a, b, r)
                            new_sum = Sums[a, b, r]
                            old_sum = new_sum - delta * u_val * v_val
                            Sums[a, b, r] = old_sum
                            if idx in S_set:
                                E_S += (old_sum - 1.0)**2 - (new_sum - 1.0)**2
                            else:
                                E_Z += old_sum**2 - new_sum**2

        return None

    # Search for lower ranks
    # Try 20, 19, 18
    for rank in range(20, 17, -1):
        if time.time() > deadline:
            break
        # Try a few restarts
        for _ in range(5):
            if time.time() > deadline:
                break
            result = local_search(rank, 20.0)
            if result:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
