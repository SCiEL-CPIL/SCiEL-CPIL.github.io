import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
DEADLINE = 150
SAFETY_MARGIN = 10

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Best result found so far (schoolbook)
    best_rank = rank_schoolbook
    best_result = None
    
    # Construct schoolbook as fallback
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        # Check that all entries are integers or half-integers
        for row in U:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in V:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in W:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        
        # Verify tensor identity: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta[i',i] * delta[j',j] * delta[k',k]
        # Actually the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if (i,j,k) corresponds to a valid triple, but wait...
        
        # The matrix multiplication tensor T is defined such that:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if this term appears in the sum
        # Actually, the standard decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # where (U ⊗ V ⊗ W)[i,j,k] = U[i] * V[j] * W[k]
        # and T[i,j,k] = 1 for all valid (i,j,k) since each product A[i,j]*B[j,k] contributes to C[i,k]
        
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def try_round_and_verify(U_float, V_float, W_float, rank):
        """Try rounding to half-integers and verify."""
        U_r = [[round(x * 2) / 2 for x in row] for row in U_float]
        V_r = [[round(x * 2) / 2 for x in row] for row in V_float]
        W_r = [[round(x * 2) / 2 for x in row] for row in W_float]
        
        if verify(U_r, V_r, W_r, rank):
            return U_r, V_r, W_r
        return None
    
    # Try to find lower rank decompositions
    # Known results: rank(2x3x3x6) is known to be at most 24 (from various sources)
    # Let's try ranks from 18 up to 24
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(18, 25):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try multiple random initializations
        max_trials = 50
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Random initialization
            U_init = rng.randn(n * m, target_rank)
            V_init = rng.randn(m * p, target_rank)
            W_init = rng.randn(p * n, target_rank)
            
            # Normalize to prevent overflow/underflow
            for c in range(target_rank):
                u_norm = np.linalg.norm(U_init[:, c])
                v_norm = np.linalg.norm(V_init[:, c])
                w_norm = np.linalg.norm(W_init[:, c])
                if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                    scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                    U_init[:, c] /= scale
                    V_init[:, c] /= scale
                    W_init[:, c] /= scale
            
            # Use gradient descent to minimize ||sum_c U_c ⊗ V_c ⊗ W_c - T||^2
            # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
            
            def compute_loss(U, V, W):
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for c in range(target_rank):
                                total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            loss += (total - 1.0) ** 2
                return loss
            
            def compute_gradients(U, V, W):
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for c in range(target_rank):
                                total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            error = 2.0 * (total - 1.0)
                            for c in range(target_rank):
                                grad_U[i * m + j, c] += error * V[j * p + k, c] * W[k * n + i, c]
                                grad_V[j * p + k, c] += error * U[i * m + j, c] * W[k * n + i, c]
                                grad_W[k * n + i, c] += error * U[i * m + j, c] * V[j * p + k, c]
                return grad_U, grad_V, grad_W
            
            # Simple gradient descent with learning rate schedule
            lr = 0.01
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            initial_loss = compute_loss(U_curr, V_curr, W_curr)
            
            max_iter = 200
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                gU, gV, gW = compute_gradients(U_curr, V_curr, W_curr)
                
                U_curr -= lr * gU
                V_curr -= lr * gV
                W_curr -= lr * gW
                
                # Decay learning rate
                lr = 0.01 * (0.995 ** it)
                
                if it % 20 == 0:
                    curr_loss = compute_loss(U_curr, V_curr, W_curr)
                    if curr_loss < 1e-6:
                        break
            
            # Try to round and verify
            result = try_round_and_verify(U_curr, V_curr, W_curr, target_rank)
            if result is not None:
                U_r, V_r, W_r = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    if best_rank <= 18:
                        break
        
        if best_rank <= 18:
            break
    
    return best_result
