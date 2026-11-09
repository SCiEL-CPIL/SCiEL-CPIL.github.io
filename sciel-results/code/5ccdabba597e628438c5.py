"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook solution
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # If time is too short, just return schoolbook
    if time.time() > deadline - 10:
        return best_result

    # Search for lower ranks
    # We use a local search on the integer grid (values are multiples of 0.5)
    # We represent values as integers multiplied by 2 to avoid floating point issues
    # during the search, then divide by 2 at the end.
    
    # Dimensions
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute indices for efficient error calculation if needed, 
    # but einsum is likely fast enough for small n,m,p,r.
    
    def calculate_error(U, V, W, r):
        """
        U: np.array (n, m, r)
        V: np.array (m, p, r)
        W: np.array (p, n, r)
        Returns sum of squared errors.
        """
        # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        # Target is 1.0 for all i,j,k
        # Error is sum (S - 1)^2
        return np.sum((S - 1.0) ** 2)

    def local_search(U, V, W, r, max_iters, deadline, temperature_start=1.0, temperature_end=0.1):
        """
        Simulated annealing on the integer grid (scaled by 2).
        U, V, W are int arrays.
        """
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        
        current_error = calculate_error(current_U.astype(float), current_V.astype(float), current_W.astype(float), r)
        
        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
        best_error = current_error
        
        temp = temperature_start
        cooling_rate = (temperature_end / temperature_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random matrix, row, col
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, nm - 1)
                # Reshape to 3D for indexing? U is (n, m, r)
                # row in nm corresponds to i*m + j
                i = row // m
                j = row % m
                c = random.randint(0, r - 1)
                mat = current_U
                idx = (i, j, c)
            elif mat_idx == 1:
                row = random.randint(0, mp - 1)
                j = row // p
                k = row % p
                c = random.randint(0, r - 1)
                mat = current_V
                idx = (j, k, c)
            else:
                row = random.randint(0, pn - 1)
                k = row // n
                i = row % n
                c = random.randint(0, r - 1)
                mat = current_W
                idx = (k, i, c)
            
            old_val = mat[idx]
            # Change by +/- 1 (which is +/- 0.5 in actual value)
            delta = random.choice([-1, 1])
            new_val = old_val + delta
            
            # Check bounds? Integers can be large, but let's keep them reasonable
            # to avoid overflow or extreme values.
            if abs(new_val) > 10:
                continue
                
            mat[idx] = new_val
            
            new_error = calculate_error(current_U.astype(float), current_V.astype(float), current_W.astype(float), r)
            
            # Acceptance probability
            if new_error < current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
            else:
                # Metropolis criterion
                prob = np.exp(-(new_error - current_error) / (temp + 1e-9))
                if random.random() < prob:
                    current_error = new_error
                else:
                    # Reject
                    mat[idx] = old_val
            
            temp *= cooling_rate
            
            # Early termination if perfect
            if current_error < 1e-9:
                break
        
        return best_U, best_V, best_W, best_error

    # Try ranks from best_rank - 1 down to 1
    # We limit the time per rank to ensure we don't spend too long on unlikely ranks
    time_per_rank = 10  # seconds
    max_restarts = 5
    
    for r in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 5:
            break
            
        rank_deadline = time.time() + time_per_rank
        
        for restart in range(max_restarts):
            if time.time() > rank_deadline:
                break
                
            # Initialize with random small integers
            # Scale: values are integers representing 0.5 * integer
            # So 0 means 0, 1 means 0.5, 2 means 1.0, etc.
            # We want to find integer/half-integer solutions.
            # The schoolbook solution uses 1.0, which is 2 in this scale.
            
            U_init = np.random.randint(-2, 3, size=(n, m, r))
            V_init = np.random.randint(-2, 3, size=(m, p, r))
            W_init = np.random.randint(-2, 3, size=(p, n, r))
            
            # Run local search
            iters = 5000
            U_out, V_out, W_out, err = local_search(U_init, V_init, W_init, r, iters, rank_deadline)
            
            if err < 1e-7:
                # Found a solution!
                # Convert back to float (divide by 2)
                U_float = (U_out.astype(float) / 2.0).tolist()
                V_float = (V_out.astype(float) / 2.0).tolist()
                W_float = (W_out.astype(float) / 2.0).tolist()
                
                # Reshape to 2D lists as required
                # U is (n, m, r) -> (n*m, r)
                U_list = []
                for i in range(n):
                    for j in range(m):
                        U_list.append(U_float[i][j])
                
                V_list = []
                for j in range(m):
                    for k in range(p):
                        V_list.append(V_float[j][k])
                        
                W_list = []
                for k in range(p):
                    for i in range(n):
                        W_list.append(W_float[k][i])
                        
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break # Move to next lower rank
        
        # If we found a solution for this rank, we continue to try lower ranks
        # If we didn't, we continue to lower ranks as well, but with less hope
        # We can break if we've spent too much time
        if time.time() > deadline - 20:
            break

    return best_result

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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of matrix multiplication."""
    rank = len(U[0])
    # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
