import numpy as np
import random
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 145  # Safety margin before 150s limit
    
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
    
    def verify_and_get_error(U, V, W, rank):
        U_np = np.array(U, dtype=float).reshape(n, m, rank)
        V_np = np.array(V, dtype=float).reshape(m, p, rank)
        W_np = np.array(W, dtype=float).reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np)
        error = np.max(np.abs(T - 1.0))
        return error, T
    
    def berlekamp_massey(sequence):
        """
        Berlekamp-Massey algorithm to find the shortest linear recurrence
        that generates the given sequence.
        
        Returns:
        - L: length of the recurrence
        - C: connection polynomial coefficients [1, c1, c2, ..., cL]
        """
        L = 0
        C = [1]  # Connection polynomial, C[0] = 1
        B = [1]  # Previous connection polynomial
        m = 1  # Number of steps since last update
        b = 1  # Previous discrepancy
        
        for n_idx in range(len(sequence)):
            # Compute discrepancy
            d = sequence[n_idx]
            for i in range(1, L + 1):
                d += C[i] * sequence[n_idx - i]
            
            if d == 0:
                m += 1
            else:
                T = C[:]
                # C = C - (d/b) * x^m * B
                factor = d / b
                for i in range(len(B)):
                    if i + m < len(C):
                        C[i + m] -= factor * B[i]
                    else:
                        # Extend C
                        C.extend([0] * (i + m - len(C) + 1))
                        C[i + m] -= factor * B[i]
                
                if 2 * L <= n_idx:
                    L = n_idx + 1 - L
                    B = T
                    b = d
                    m = 1
                else:
                    m += 1
        
        return L, C
    
    def hankel_determinants_from_flattening(matrix, size):
        """
        Compute Hankel determinants from a 1D sequence.
        For a sequence s, the k-th Hankel determinant is:
        det([[s_0, s_1, ..., s_{k-1}],
             [s_1, s_2, ..., s_k],
             ...,
             [s_{k-1}, s_k, ..., s_{2k-1}]])
        """
        det = 0
        # This is a simplified version - in practice we'd use the actual Hankel matrix
        # For now, we'll use a different approach based on the sequence directly
        
        # Actually, let's use the standard BM approach on the sequence itself
        pass
    
    def greedy_polish(U, V, W, rank, max_iters=500):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        current_err, _ = verify_and_get_error(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank)
        
        if current_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist(), 0.0
        
        values = [-2, -1, 0, 1, 2]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
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
    
    # Use Berlekamp-Massey to estimate minimal rank
    # The tensor T[i,j,k] = 1 for all i,j,k can be viewed as a sequence
    # We'll try to find a low-rank decomposition using BM on related sequences
    
    # Strategy: For each candidate rank r, try to construct U,V,W with that rank
    # Use BM to guide the construction
    
    # First, let's try ranks from a reasonable lower bound
    # Lower bound for 2x3x7 matrix multiplication is at least max(2*3, 3*7, 2*7) = 21
    # But we know schoolbook is 42, so try between 21 and 42
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 2000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
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
            if err < 0.5:
                U_pol, V_pol, W_pol, final_err = greedy_polish(U, V, W, target_rank)
                if final_err < 1e-9:
                    best = {"U": U_pol, "V": V_pol, "W": W_pol}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    return best
