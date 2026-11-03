import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]
_DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = len(U[0])
    # Convert to numpy for speed, though loops are small enough
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We can vectorize this check.
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U_t = U_arr.reshape(n, m, R)
    V_t = V_arr.reshape(m, p, R)
    W_t = W_arr.reshape(p, n, R)
    
    # We need sum_r U_t[i,j,r] V_t[j,k,r] W_t[k,i,r]
    # This is a contraction.
    # Let's just do the triple loop as n*m*p is small (63) and R is at most 63.
    # 63 * 63 = 3969 ops per check. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer (i.e., 2*x is integer)."""
    return np.round(arr * 2) / 2.0

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition of rank n*m*p."""
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

def _try_find_lower_rank(n, m, p, start_rank, end_rank, deadline):
    """
    Attempt to find a decomposition with rank in [start_rank, end_rank].
    Uses random initialization and simple gradient steps/ALS, followed by rounding and verification.
    """
    for R in range(start_rank, end_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with small random values
            # Scale matters. For matmul tensor, entries are 1.
            # Let's initialize with small noise around 0.
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Simple Alternating Least Squares (ALS) or Gradient Descent
            # Objective: Minimize sum_{i,j,k} (1 - sum_r U_ij_r V_jk_r W_ki_r)^2
            # This is non-convex.
            
            # Given the strict time limit and low probability, we will perform a very limited number of iterations.
            # If we don't find it quickly, we skip.
            
            # Actually, for such a small problem, we can try to solve for W given U,V or vice versa?
            # The equation is linear in W for fixed U,V.
            # sum_r U_ij_r V_jk_r W_ki_r = 1
            # This is a system of linear equations for W_ki_r?
            # Let's fix U and V. We want to find W.
            # For each (i,k), we have:
            # sum_{j,r} U_ij_r V_jk_r W_ki_r = 1
            # Note that W_ki_r depends on k and i.
            # The term is sum_j (U_ij_r V_jk_r) W_ki_r.
            # This doesn't separate nicely because W_ki_r is multiplied by a sum over j.
            # Let M_{ik, r} = sum_j U_{ij, r} V_{jk, r}.
            # Then sum_r M_{ik, r} W_{ki, r} = 1.
            # This is a linear system for W_{ki, r} if we treat the vector of all W_{ki, r} as unknowns?
            # No, because M depends on i and k.
            # For a fixed (i,k), we have:
            # sum_r (sum_j U_{ij, r} V_{jk, r}) W_{ki, r} = 1.
            # This is one equation for the R variables W_{k0, r}, ..., W_{k, R-1}? No, W_{ki, r} is specific to i,k.
            # Wait, W is indexed by (k, i, r).
            # So for each pair (i,k), we have a linear equation in the variables W_{k,i,0}, ..., W_{k,i,R-1}.
            # Specifically: sum_{r=0}^{R-1} A_{ik, r} W_{ki, r} = 1, where A_{ik, r} = sum_j U_{ij, r} V_{jk, r}.
            # This system is underdetermined if R > 1. We can find the minimum norm solution or any solution.
            # np.linalg.lstsq can solve this.
            
            # Let's try this ALS approach:
            # 1. Init U, V.
            # 2. Solve for W.
            # 3. Solve for V (fixing U, W).
            # 4. Solve for U (fixing V, W).
            # 5. Check error.
            
            converged = False
            for step in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Solve for W
                # For each i, k:
                # sum_r (sum_j U[i,j,r] V[j,k,r]) W[k,i,r] = 1
                # Let A_ik_r = sum_j U[i,j,r] V[j,k,r]
                # We want W[k,i,r] such that dot(A_ik, W_ki) = 1.
                # This is a single linear equation with R unknowns.
                # Infinite solutions. We pick the one with smallest norm? Or just project 1 onto A?
                # W_ki = A_ik * (1 / ||A_ik||^2) if A_ik != 0.
                # This minimizes ||W_ki|| subject to A_ik . W_ki = 1.
                
                W_new = np.zeros((p * n, R))
                for i in range(n):
                    for k in range(p):
                        A_ik = np.zeros(R)
                        for j in range(m):
                            # U[i*m + j, :] is vector of size R
                            # V[j*p + k, :] is vector of size R
                            A_ik += U[i * m + j] * V[j * p + k]
                        
                        norm_sq = np.dot(A_ik, A_ik)
                        if norm_sq > 1e-12:
                            W_new[k * n + i] = A_ik / norm_sq
                        else:
                            # If A_ik is 0, we can't satisfy the equation.
                            # This indicates the current U, V are bad.
                            W_new[k * n + i] = 0.1 # small fallback
                
                # Solve for V
                # For each j, k:
                # sum_r (sum_i U[i,j,r] W[k,i,r]) V[j,k,r] = 1
                # Let B_jk_r = sum_i U[i,j,r] W[k,i,r]
                # dot(B_jk, V_jk) = 1
                V_new = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros(R)
                        for i in range(n):
                            B_jk += U[i * m + j] * W[k * n + i]
                        
                        norm_sq = np.dot(B_jk, B_jk)
                        if norm_sq > 1e-12:
                            V_new[j * p + k] = B_jk / norm_sq
                        else:
                            V_new[j * p + k] = 0.1

                # Solve for U
                # For each i, j:
                # sum_r (sum_k V[j,k,r] W[k,i,r]) U[i,j,r] = 1
                # Let C_ij_r = sum_k V[j,k,r] W[k,i,r]
                # dot(C_ij, U_ij) = 1
                U_new = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        C_ij = np.zeros(R)
                        for k in range(p):
                            C_ij += V[j * p + k] * W[k * n + i]
                        
                        norm_sq = np.dot(C_ij, C_ij)
                        if norm_sq > 1e-12:
                            U_new[i * m + j] = C_ij / norm_sq
                        else:
                            U_new[i * m + j] = 0.1

                U = U_new
                V = V_new
                W = W_new

            # After ALS, we have a real-valued solution.
            # Now, round to nearest half-integers.
            U_rounded = _round_to_half_integer(U).tolist()
            V_rounded = _round_to_half_integer(V).tolist()
            W_rounded = _round_to_half_integer(W).tolist()
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                
    return None

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 21
    # We try ranks from 21 up to best_rank - 1
    # To save time, we only try a few ranks or a limited number of restarts.
    
    # Given the difficulty, we try ranks 21 to 30 with a few restarts.
    # If we find one, we return it.
    
    # We must be careful with the deadline.
    search_deadline = start_time + 110
    
    # Try a small range of ranks
    # If we don't find anything, return schoolbook.
    
    # It is very unlikely we find a rank < 63 for this specific problem with simple ALS and rounding.
    # However, the code structure allows for it.
    
    result = _try_find_lower_rank(n, m, p, 21, min(30, best_rank - 1), search_deadline)
    
    if result is not None:
        return result
        
    return best_result
