import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Try to find rank-5 decomposition for 2x3x3 block
    # T_block size 6x9x6
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for l in range(3):
                T_block[i*3+j, j*3+l, l*2+i] = 1.0
                
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    found_block = False
    U_block = None
    V_block = None
    W_block = None
    
    # Search for rank 5 in 2x3x3
    # Use a longer deadline to allow more search
    deadline = time.time() + 100
    
    # Try different rank values, starting from 5
    for r in range(5, 19):
        if time.time() > deadline:
            break
            
        # Try multiple random seeds for this rank
        for trial in range(200):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 1000 + trial)
            
            # Random initialization with better structure
            # Initialize U, V, W with small random values
            U = np.random.randn(6, r)
            V = np.random.randn(9, r)
            W = np.random.randn(6, r)
            
            # Normalize columns to unit norm
            U_norms = np.linalg.norm(U, axis=0, keepdims=True)
            V_norms = np.linalg.norm(V, axis=0, keepdims=True)
            W_norms = np.linalg.norm(W, axis=0, keepdims=True)
            
            # Avoid division by zero
            U_norms[U_norms < 1e-10] = 1.0
            V_norms[V_norms < 1e-10] = 1.0
            W_norms[W_norms < 1e-10] = 1.0
            
            U /= U_norms
            V /= V_norms
            W /= W_norms
            
            # ALS iterations with convergence check
            prev_error = float('inf')
            for it in range(50):
                # Solve for U
                # T_block[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W. For each a: T_block[a, :, :].flatten() = sum_r U[a,r] (V[:,r] * W[:,r]).flatten()
                M = (V[:, None, :] * W[None, :, :]).reshape(54, r)
                try:
                    U, _, _, _ = np.linalg.lstsq(M.T, T_block.reshape(6, 54).T, rcond=None)
                    U = U.T
                except:
                    break
                
                # Solve for V
                # Fix U, W. For each b: T_block[:, b, :].flatten() = sum_r V[b,r] (U[:,r] * W[:,r]).flatten()
                N = (U[:, None, :] * W[None, :, :]).reshape(36, r)
                try:
                    V, _, _, _ = np.linalg.lstsq(N.T, T_block.reshape(6, 9, 6).transpose(1, 0, 2).reshape(9, 36).T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Solve for W
                # Fix U, V. For each c: T_block[:, :, c].flatten() = sum_r W[c,r] (U[:,r] * V[:,r]).flatten()
                P = (U[:, None, :] * V[None, :, :]).reshape(54, r)
                try:
                    W, _, _, _ = np.linalg.lstsq(P.T, T_block.reshape(6, 9, 6).transpose(2, 0, 1).reshape(6, 54).T, rcond=None)
                    W = W.T
                except:
                    break
                
                # Check convergence
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_approx - T_block)
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            # Check if we found a valid solution
            if verify_solution(U, V, W, T_block):
                # Try to round to half-integers
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    U_rounded = np.round(2.0 * U) / 2.0
                    V_rounded = np.round(2.0 * V) / 2.0
                    W_rounded = np.round(2.0 * W) / 2.0
                    if verify_solution(U_rounded, V_rounded, W_rounded, T_block):
                        U_block = U_rounded
                        V_block = V_rounded
                        W_block = W_rounded
                        found_block = True
                        break
                else:
                    # Try to find a nearby half-integer solution using local search
                    # This is a heuristic approach
                    pass
        if found_block:
            break

    if found_block:
        # Construct full 2x3x9 decomposition
        rank_full = 3 * 5
        U_full = np.zeros((6, rank_full))
        V_full = np.zeros((27, rank_full))
        W_full = np.zeros((18, rank_full))
        
        for l in range(3):
            for r_idx in range(5):
                R = l * 5 + r_idx
                U_full[:, R] = U_block[:, r_idx]
                
                for j in range(3):
                    for s in range(3):
                        k = 3*l + s
                        V_full[j*9 + k, R] = V_block[j*3 + s, r_idx]
                        
                for s in range(3):
                    for i in range(2):
                        k = 3*l + s
                        W_full[k*2 + i, R] = W_block[s*2 + i, r_idx]
                        
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = 15
    else:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 54

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
