import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    deadline = time.time() + 115  # Safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute valid triples (u_idx, v_idx, w_idx)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
                
    num_triples = len(triples)
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for (u_idx, v_idx, w_idx) in triples:
            s = 0.0
            for c in range(rank):
                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
            if abs(s - 1.0) > 1e-9:
                return False
        return True

    def to_half_int(x):
        return round(x * 2) / 2

    def try_rank_als(rank, max_restarts=20, max_iter=100):
        """
        Use Alternating Least Squares to find a rank-r decomposition.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random Initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize to prevent numerical issues
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # 1. Solve for U
                # Variables: U_flat (size nm * rank)
                # Equations: num_triples
                # A[row, u*rank + c] = V[v, c] * W[w, c]
                A_u = np.zeros((num_triples, nm * rank))
                for idx, (u_idx, v_idx, w_idx) in enumerate(triples):
                    for c in range(rank):
                        A_u[idx, u_idx * rank + c] = V[v_idx, c] * W[w_idx, c]
                
                b = np.ones(num_triples)
                try:
                    U_flat, residuals, rank_u, s = np.linalg.lstsq(A_u, b, rcond=None)
                    U_new = U_flat.reshape(nm, rank)
                except np.linalg.LinAlgError:
                    break
                
                # 2. Solve for V
                A_v = np.zeros((num_triples, mp * rank))
                for idx, (u_idx, v_idx, w_idx) in enumerate(triples):
                    for c in range(rank):
                        A_v[idx, v_idx * rank + c] = U_new[u_idx, c] * W[w_idx, c]
                
                try:
                    V_flat, residuals, rank_v, s = np.linalg.lstsq(A_v, b, rcond=None)
                    V_new = V_flat.reshape(mp, rank)
                except np.linalg.LinAlgError:
                    break
                
                # 3. Solve for W
                A_w = np.zeros((num_triples, pn * rank))
                for idx, (u_idx, v_idx, w_idx) in enumerate(triples):
                    for c in range(rank):
                        A_w[idx, w_idx * rank + c] = U_new[u_idx, c] * V_new[v_idx, c]
                
                try:
                    W_flat, residuals, rank_w, s = np.linalg.lstsq(A_w, b, rcond=None)
                    W_new = W_flat.reshape(pn, rank)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Calculate Error
                error = 0.0
                for (u_idx, v_idx, w_idx) in triples:
                    s_val = 0.0
                    for c in range(rank):
                        s_val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    error += (s_val - 1.0) ** 2
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-12:
                    break
            
            # Check if we have a good solution
            if best_error < 1e-8:
                # Round to half-integers
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                # Verify rounded solution
                if verify(U_rounded, V_rounded, W_rounded, rank):
                    U_list = [[float(U_rounded[i][c]) for c in range(rank)] for i in range(nm)]
                    V_list = [[float(V_rounded[i][c]) for c in range(rank)] for i in range(mp)]
                    W_list = [[float(W_rounded[i][c]) for c in range(rank)] for i in range(pn)]
                    return (U_list, V_list, W_list)
        
        return None

    # Try ranks from low to high
    # For 2x3x5, known upper bounds suggest rank <= 28
    # Try a range of ranks with more restarts for lower ranks
    for rank in range(10, 30):
        if time.time() > deadline:
            break
        
        # More restarts for lower ranks (harder to find)
        max_restarts = 50 if rank <= 15 else 30 if rank <= 20 else 20
        result = try_rank_als(rank, max_restarts=max_restarts, max_iter=150)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook (rank 30)
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
