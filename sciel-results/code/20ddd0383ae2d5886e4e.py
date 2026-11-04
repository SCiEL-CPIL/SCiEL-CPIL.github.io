import numpy as np
import random
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # Safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Precompute reshaped target indices? No, we use einsum.
    
    def verify_and_get_error(U, V, W, rank):
        U_np = np.array(U, dtype=float).reshape(n, m, rank)
        V_np = np.array(V, dtype=float).reshape(m, p, rank)
        W_np = np.array(W, dtype=float).reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np)
        
        # Target is 1 for all i,j,k
        error = np.max(np.abs(T - 1.0))
        return error, T

    def greedy_polish(U, V, W, rank, max_iters=1000):
        # Convert to numpy for fast updates? 
        # Updating numpy arrays is fast, but we need to keep track of integer/half-integer constraints.
        # Let's work with integers * 2 to allow half-integers.
        # Or just integers first.
        
        # We'll use a list of lists for simplicity in updates, but check error with numpy.
        # This is a trade-off. 
        # Actually, let's use numpy arrays for U, V, W in the polish loop.
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        current_err, _ = verify_and_get_error(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank)
        
        # If current error is already 0, return
        if current_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist(), 0.0

        # Define possible values. Integers: -2, -1, 0, 1, 2
        # Half-integers: -1.5, -0.5, 0.5, 1.5
        # Let's stick to integers for now to keep it simple and fast.
        # If we need half-integers, we can scale.
        
        values = [-2, -1, 0, 1, 2]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2:
            improved = False
            iters += 1
            
            # Try updating U
            for i in range(nm):
                for c in range(rank):
                    old_val = U_np[i][c]
                    for new_val in values:
                        if new_val == old_val: continue
                        U_np[i][c] = new_val
                        err, _ = verify_and_get_error(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank)
                        if err < current_err - 1e-9:
                            current_err = err
                            improved = True
                        else:
                            U_np[i][c] = old_val
                            
            if current_err < 1e-9:
                break

            # Try updating V
            for i in range(mp):
                for c in range(rank):
                    old_val = V_np[i][c]
                    for new_val in values:
                        if new_val == old_val: continue
                        V_np[i][c] = new_val
                        err, _ = verify_and_get_error(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank)
                        if err < current_err - 1e-9:
                            current_err = err
                            improved = True
                        else:
                            V_np[i][c] = old_val
                            
            if current_err < 1e-9:
                break

            # Try updating W
            for i in range(pn):
                for c in range(rank):
                    old_val = W_np[i][c]
                    for new_val in values:
                        if new_val == old_val: continue
                        W_np[i][c] = new_val
                        err, _ = verify_and_get_error(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank)
                        if err < current_err - 1e-9:
                            current_err = err
                            improved = True
                        else:
                            W_np[i][c] = old_val
                            
            if current_err < 1e-9:
                break

        return U_np.tolist(), V_np.tolist(), W_np.tolist(), current_err

    # Search for lower rank
    # Lower bound is roughly 21.
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 5000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with integers -1, 0, 1
            # Or -2, -1, 1, 2
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(pn)]
            
            err, T = verify_and_get_error(U, V, W, target_rank)
            
            if err < 1e-9:
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
            
            # If error is small, try to polish
            if err < 1.0:
                U_pol, V_pol, W_pol, final_err = greedy_polish(U, V, W, target_rank)
                if final_err < 1e-9:
                    best = {"U": U_pol, "V": V_pol, "W": W_pol}
                    best_rank = target_rank
                    found = True
                    break
                # If polish improved it significantly, maybe keep it? 
                # But we only accept exact solutions.
        
        if found:
            continue

    return best
