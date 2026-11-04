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
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
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

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_linear_solve_rank(target_rank, max_attempts):
        """
        Try to find a decomposition of rank R by fixing U and V randomly
        and solving for W linearly.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random sparse U and V
            # Using {-1, 0, 1} to allow for cancellations
            U = np.zeros((nm, target_rank), dtype=np.int64)
            V = np.zeros((mp, target_rank), dtype=np.int64)
            
            # Randomly populate with 1s and -1s
            # Density around 0.5
            for c in range(target_rank):
                for i in range(nm):
                    if random.random() < 0.5:
                        U[i, c] = 1 if random.random() < 0.5 else -1
                for j in range(mp):
                    if random.random() < 0.5:
                        V[j, c] = 1 if random.random() < 0.5 else -1
            
            # Construct linear system M * W_flat = b
            # M has shape (n*m*p, pn * target_rank)
            # b has shape (n*m*p,)
            # M[(i,j,k), (k,i,c)] = U[i,j,c] * V[j,k,c]
            
            M = np.zeros((n*m*p, pn * target_rank), dtype=np.float64)
            b = np.ones(n*m*p, dtype=np.float64)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            u_val = U[i*m + j, c]
                            v_val = V[j*p + k, c]
                            m_val = u_val * v_val
                            if m_val != 0:
                                # W index is (k, i, c) -> flat index k*n + i, then * target_rank + c
                                w_row = k * n + i
                                w_flat_idx = w_row * target_rank + c
                                M[idx, w_flat_idx] = m_val
                        idx += 1
            
            # Solve M W = b
            # Use least squares to find a solution
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check if the solution is close to integer or half-integer
            # And check the residual
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-6:
                    continue
            
            # Check integrality
            is_valid = True
            W_list = []
            for val in W_flat:
                # Check if close to integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                
                if abs(val - nearest_int) < 1e-6:
                    W_list.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    W_list.append(nearest_half) # Store as float for now, convert later if needed
                    # Actually, the problem allows half-integers.
                    # Let's keep them as floats in the list, but ensure they are .0 or .5
                else:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Reshape W_flat to (pn, target_rank)
            W_np = np.array(W_list).reshape(pn, target_rank)
            
            # Convert to list of lists
            # The problem statement says "integer/half-integer entries".
            # If we use floats, we need to ensure they are exact.
            # Let's verify with the original verify function.
            # But verify_fast uses int64. We need to handle half-integers.
            # If we have half-integers, we can multiply everything by 2?
            # No, the decomposition must be exact.
            # If W has 0.5, then U and V should be integers.
            # Let's just pass the lists to a verifier that handles floats.
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list_of_lists = W_np.tolist()
            
            # Verify
            U_np_v = np.array(U_list, dtype=np.float64)
            V_np_v = np.array(V_list, dtype=np.float64)
            W_np_v = np.array(W_list_of_lists, dtype=np.float64)
            
            U3 = U_np_v.reshape(n, m, target_rank)
            V3 = V_np_v.reshape(m, p, target_rank)
            W3 = W_np_v.reshape(p, n, target_rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            if np.allclose(T, 1.0, atol=1e-6):
                # Ensure exactness by rounding to nearest 0.5
                # If it passes close, it's likely exact due to linear solve
                return {
                    "U": U_list,
                    "V": V_list,
                    "W": W_list_of_lists
                }
                
        return None

    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = 21
    # We iterate from low to high.
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # More attempts for lower ranks? Or just fixed attempts?
        # Linear solve is fast, so we can do more attempts.
        max_attempts = 5000 if target_rank < 30 else 1000
        
        result = try_linear_solve_rank(target_rank, max_attempts)
        if result:
            best = result
            best_rank = target_rank
            break

    return best
# EVOLVE-BLOCK-END
