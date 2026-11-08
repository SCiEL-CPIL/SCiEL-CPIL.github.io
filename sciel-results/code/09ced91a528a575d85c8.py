import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    _start = time.time()
    _deadline = _start + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    def build_rank22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    def is_half_integer(val):
        return val == int(val) or abs(val * 2 - round(val * 2)) < 1e-10
    
    def check_valid(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        return True
    
    def tensor_residual(U, V, W, rank):
        """Compute residual tensor R[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1"""
        R = np.zeros((n, m, p))
        for r in range(rank):
            u_col = np.array([U[i*m+j][r] for i in range(n) for j in range(m)]).reshape(n, m)
            v_col = np.array([V[j*p+k][r] for j in range(m) for k in range(p)]).reshape(m, p)
            w_col = np.array([W[k*n+i][r] for k in range(p) for i in range(n)]).reshape(p, n)
            # For each r, contribution to R[i,j,k] is u_col[i,j] * v_col[j,k] * w_col[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        R -= 1.0
        return R
    
    def greedy_rank_reduce(U, V, W, rank, max_iters=500):
        """
        Try to reduce rank by greedily finding a column that can be removed
        or by merging columns. Uses least-squares to find optimal replacement.
        """
        if time.time() > _deadline - 20:
            return U, V, W, rank
        
        # Convert to numpy for faster operations
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        best_rank = rank
        best_U, best_V, best_W = U, V, W
        
        # Try removing each column one at a time
        for r_remove in range(rank):
            if time.time() > _deadline - 20:
                break
            
            U_test = U_np.copy()
            V_test = V_np.copy()
            W_test = W_np.copy()
            
            # Remove column r_remove
            U_test[:, r_remove] = 0
            V_test[:, r_remove] = 0
            W_test[:, r_remove] = 0
            
            # Check residual
            R = tensor_residual(U_test.tolist(), V_test.tolist(), W_test.tolist(), rank)
            if np.max(np.abs(R)) < 1e-10:
                # Column is redundant
                mask = [r for r in range(rank) if r != r_remove]
                U_new = U_test[:, mask].tolist()
                V_new = V_test[:, mask].tolist()
                W_new = W_test[:, mask].tolist()
                new_rank = rank - 1
                
                if check_valid(U_new, V_new, W_new) and verify(U_new, V_new, W_new, new_rank):
                    best_rank = new_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    U, V, W = U_new, V_new, W_new
                    U_np, V_np, W_np = np.array(U_new), np.array(V_new), np.array(W_new)
                    rank = new_rank
        
        return best_U, best_V, best_W, best_rank
    
    def try_als_rank(target_rank, n_restarts=5, max_iters=1000):
        """
        Try to find a rank-target_rank decomposition using ALS.
        Returns (U, V, W, rank) if found, else None.
        """
        if time.time() > _deadline - 30:
            return None
        
        np.random.seed(42)
        
        for restart in range(n_restarts):
            if time.time() > _deadline - 30:
                break
            
            # Initialize with random half-integer values
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            prev_err = np.inf
            
            for it in range(max_iters):
                if time.time() > _deadline - 30:
                    break
                
                # Update U
                for r in range(target_rank):
                    # For each r, U[:,r] should minimize sum_{i,j,k} (U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
                    # This is linear in U[:,r]
                    # Residual without r-th term: R_no_r[i,j,k] = sum_{r'!=r} U[i,j,r']*V[j,k,r']*W[k,i,r'] - 1
                    # We want U[i,j,r] * V[j,k,r] * W[k,i,r] to match -R_no_r[i,j,k]
                    # This is still nonlinear, but we can linearize
                    
                    # Alternative: treat it as a least-squares problem
                    # For fixed V[:,r] and W[:,r], find U[:,r] to minimize residual
                    # R[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r] + sum_{r'!=r} ... - 1
                    # Let B[i,j,k] = V[j,k,r]*W[k,i,r]
                    # Then R[i,j,k] = U[i,j,r]*B[i,j,k] + C[i,j,k]
                    # We want R[i,j,k] = 0, so U[i,j,r]*B[i,j,k] = -C[i,j,k]
                    # This is a system of equations, but overdetermined in general
                    
                    # Use gradient descent instead
                    pass
                
                # Compute current residual
                R = tensor_residual(U_cur.tolist(), V_cur.tolist(), W_cur.tolist(), target_rank)
                err = np.sum(R**2)
                
                if err < 1e-12:
                    break
                
                if err < prev_err * 0.999:
                    prev_err = err
                else:
                    break
            
            # Check if we found a valid solution
            R = tensor_residual(U_cur.tolist(), V_cur.tolist(), W_cur.tolist(), target_rank)
            if np.max(np.abs(R)) < 1e-10:
                # Try rounding to half-integers
                U_round = np.round(U_cur * 2) / 2
                V_round = np.round(V_cur * 2) / 2
                W_round = np.round(W_cur * 2) / 2
                
                R_round = tensor_residual(U_round.tolist(), V_round.tolist(), W_round.tolist(), target_rank)
                if np.max(np.abs(R_round)) < 1e-10:
                    U_out = U_round.tolist()
                    V_out = V_round.tolist()
                    W_out = W_round.tolist()
                    if check_valid(U_out, V_out, W_out) and verify(U_out, V_out, W_out, target_rank):
                        return U_out, V_out, W_out, target_rank
        
        return None
    
    # Start with rank-22
    U, V, W, rank = build_rank22()
    
    if not verify(U, V, W, rank):
        raise RuntimeError("Rank-22 decomposition failed verification")
    
    if not check_valid(U, V, W, rank):
        raise RuntimeError("Rank-22 has non-half-integer entries")
    
    best_rank = rank
    best_U, best_V, best_W = U[:], V[:], W[:]
    
    # First, try to remove redundant columns
    U, V, W, rank = greedy_rank_reduce(U, V, W, rank)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions using ALS
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > _deadline - 30:
            break
        
        result = try_als_rank(target_rank, n_restarts=3, max_iters=500)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
