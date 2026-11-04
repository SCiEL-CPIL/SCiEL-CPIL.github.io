"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_solve_for_W(U, V, rank):
        """
        Given U and V, solve for W.
        Returns W if integer solution found, else None.
        """
        W = np.zeros((pn, rank), dtype=np.int64)
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        
        for i in range(n):
            for k in range(p):
                # M[j, c] = U[i,j,c] * V[j,k,c]
                M = U3[i, :, k] * V3[:, k, :]
                
                # Solve M w = 1
                try:
                    w_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check if w_sol is close to integers
                w_int = np.round(w_sol)
                
                # Verify if M @ w_int == 1
                if not np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                    return None
                
                W[k*n + i, :] = w_int
        
        # Final verification
        if verify_fast(U.tolist(), V.tolist(), W.tolist(), rank):
            return W.tolist()
        return None

    def try_random_rank_solve(target_rank, max_attempts, value_range):
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            nnz_u = 2
            nnz_v = 2
            
            U = np.zeros((nm, target_rank), dtype=np.int64)
            V = np.zeros((mp, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                u_rows = random.sample(range(nm), min(nnz_u, nm))
                v_rows = random.sample(range(mp), min(nnz_v, mp))
                
                for r in u_rows:
                    U[r, c] = random.randint(value_range[0], value_range[1])
                for r in v_rows:
                    V[r, c] = random.randint(value_range[0], value_range[1])
            
            W_sol = try_solve_for_W(U, V, target_rank)
            if W_sol:
                return {"U": U.tolist(), "V": V.tolist(), "W": W_sol}
        
        return None

    # Also keep the original sparse random search as a backup if solve-for-W fails
    def try_sparse_random_rank(target_rank, max_attempts, value_range):
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            nnz_per_col = 2
            
            U = np.zeros((nm, target_rank), dtype=np.int64)
            V = np.zeros((mp, target_rank), dtype=np.int64)
            W = np.zeros((pn, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                u_rows = random.sample(range(nm), min(nnz_per_col, nm))
                v_rows = random.sample(range(mp), min(nnz_per_col, mp))
                w_rows = random.sample(range(pn), min(nnz_per_col, pn))
                
                for r in u_rows:
                    U[r, c] = random.randint(value_range[0], value_range[1])
                for r in v_rows:
                    V[r, c] = random.randint(value_range[0], value_range[1])
                for r in w_rows:
                    W[r, c] = random.randint(value_range[0], value_range[1])
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try solve-for-W first
        result = try_random_rank_solve(target_rank, 300, (-2, 2))
        
        if not result:
            # Try sparse random as backup
            result = try_sparse_random_rank(target_rank, 100, (-2, 2))
        
        if result:
            best = result
            best_rank = target_rank
            if target_rank == 21:
                break
                
    return best
# EVOLVE-BLOCK-END
