"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=120.0):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    Uses a randomized approach to eliminate columns if they are linearly dependent
    in the tensor product space.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    current_rank = len(U[0])
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    # We will try to eliminate columns one by one.
    # For each column c, we check if the tensor contribution of c can be expressed
    # as a linear combination of the other columns.
    # This is equivalent to solving for alpha_d such that:
    # sum_{d != c} alpha_d * U[:,d] * V[:,d] * W[:,d] = U[:,c] * V[:,c] * W[:,c]
    
    # This is a linear system for alpha_d.
    # Number of equations: n*m*p = 48
    # Number of unknowns: current_rank - 1
    
    # We iterate until we can't reduce further or time runs out.
    while time.time() < deadline - 2.0:
        if current_rank <= 1:
            break
            
        # Try to eliminate a random column
        c_elim = random.randint(0, current_rank - 1)
        
        num_eqs = n * m * p
        num_unk = current_rank - 1
        
        # Build the linear system A * alpha = b
        A = np.zeros((num_eqs, num_unk))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target value for this equation is the contribution of column c_elim
                    target = U_np[i*m+j, c_elim] * V_np[j*p+k, c_elim] * W_np[k*n+i, c_elim]
                    b[eq_idx] = target
                    
                    unk_idx = 0
                    for d in range(current_rank):
                        if d != c_elim:
                            A[eq_idx, unk_idx] = U_np[i*m+j, d] * V_np[j*p+k, d] * W_np[k*n+i, d]
                            unk_idx += 1
                    eq_idx += 1
        
        # Solve the linear system
        try:
            # Use least squares if the system is overdetermined (which it isn't here, but safe)
            # Or just np.linalg.solve if square/invertible
            # Since num_unk (r-1) is usually > num_eqs (48) for r > 49, it's underdetermined.
            # For r <= 49, it might be square or overdetermined.
            # We want to find *any* solution alpha.
            
            # If underdetermined, use lstsq which finds minimum norm solution
            alpha, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is valid (residuals should be close to 0)
            # Note: residuals from lstsq is the sum of squared residuals
            if len(residuals) > 0:
                res_sq = residuals[0]
            else:
                res_sq = np.sum((A @ alpha - b)**2)
                
            if res_sq < 1e-6:
                # We found a valid dependency!
                # We can eliminate column c_elim.
                # The new decomposition is:
                # T = sum_{d != c_elim} (U_d * V_d * W_d) * alpha_d + U_c * V_c * W_c * 1
                # Wait, the equation was:
                # sum_{d != c} alpha_d T_d = T_c
                # So T_c = sum_{d != c} alpha_d T_d
                # Thus sum_{all} T_d = sum_{d != c} T_d + T_c = sum_{d != c} T_d + sum_{d != c} alpha_d T_d
                # = sum_{d != c} (1 + alpha_d) T_d
                # So we can just scale the existing columns by (1 + alpha_d) and remove c_elim.
                
                new_U = []
                new_V = []
                new_W = []
                
                alpha_idx = 0
                for d in range(current_rank):
                    if d != c_elim:
                        scale = 1.0 + alpha[alpha_idx]
                        new_U.append(U_np[:, d] * scale)
                        new_V.append(V_np[:, d] * scale)
                        new_W.append(W_np[:, d] * scale)
                        alpha_idx += 1
                        
                new_U = np.column_stack(new_U)
                new_V = np.column_stack(new_V)
                new_W = np.column_stack(new_W)
                
                # Verify the new decomposition
                # We need to convert back to list of lists for the return format eventually,
                # but let's verify with numpy first for speed
                # T_new[a,b,c] = sum_l new_U[a,l] new_V[b,l] new_W[c,l]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for l in range(len(new_U[0])):
                                total += new_U[i*m+j, l] * new_V[j*p+k, l] * new_W[k*n+i, l]
                            if abs(total - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    current_rank -= 1
                    U_np = new_U
                    V_np = new_V
                    W_np = new_W
                    best_rank = current_rank
                    # Convert to lists for storage? Or keep numpy?
                    # The function returns U, V, W as lists of lists.
                    # Let's keep updating the numpy arrays and convert at the end if successful.
                    # But we need to update the 'best' variables.
                    best_U = U_np.tolist()
                    best_V = V_np.tolist()
                    best_W = W_np.tolist()
                    
        except np.linalg.LinAlgError:
            pass
        
        # If we reduced rank, we might want to try reducing further.
        # If not, we might want to try a different column to eliminate.
        # To avoid getting stuck, we can randomize the order or just continue.
        
    # Convert back to list of lists
    if best_rank < current_rank:
        # best_U, best_V, best_W are already lists if we updated them
        return best_U, best_V, best_W, best_rank
    else:
        return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, max_time=110.0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
