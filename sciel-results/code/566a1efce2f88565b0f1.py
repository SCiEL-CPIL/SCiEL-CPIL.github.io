"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48 schoolbook
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    # Schoolbook decomposition
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    def verify_vectorized(U, V, W, r, field=None):
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_r = U[:, :r].reshape(n, m, r)
        V_r = V[:, :r].reshape(m, p, r)
        W_r = W[:, :r].reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note: W is indexed by k*n+i, so W_r[k, i, c]
        # einsum: 'ijc,jkc,kic->ijk'
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        except Exception:
            return False

        if field:
            T = T % field
            return np.all(T == 1)
        else:
            return np.all(T == 1)

    # Helper to return a valid solution
    def make_solution(U, V, W, r):
        return {
            "U": U[:, :r].astype(int).tolist(),
            "V": V[:, :r].astype(int).tolist(),
            "W": W[:, :r].astype(int).tolist()
        }

    # Try to reduce rank. 
    # We start with rank 47. 
    # The search space is huge. We use a very limited random search with vectorized verification.
    # If we don't find a solution quickly, we return rank 48.
    
    # We only try rank 47 and 46 to save time, as 48 is the safe fallback.
    for target_rank in [47, 46]:
        if time.time() > deadline:
            break
            
        # Try over small prime fields for easier integer solutions
        for field in [2, 3]:
            if time.time() > deadline:
                break
                
            Uf = np.zeros((n * m, target_rank), dtype=int)
            Vf = np.zeros((m * p, target_rank), dtype=int)
            Wf = np.zeros((p * n, target_rank), dtype=int)
            
            # Initialize with random values? Or structured?
            # Random initialization is better for exploration.
            Uf[:] = np.random.randint(0, field, (n * m, target_rank))
            Vf[:] = np.random.randint(0, field, (m * p, target_rank))
            Wf[:] = np.random.randint(0, field, (p * n, target_rank))
            
            # Limited number of restarts/steps
            max_restarts = 20
            max_steps = 500
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Reset to random
                Uf[:] = np.random.randint(0, field, (n * m, target_rank))
                Vf[:] = np.random.randint(0, field, (m * p, target_rank))
                Wf[:] = np.random.randint(0, field, (p * n, target_rank))
                
                for step in range(max_steps):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random entry to modify
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Uf[row, col]
                        Uf[row, col] = random.randint(0, field - 1)
                    elif mat == 'V':
                        row = random.randint(0, m * p - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Vf[row, col]
                        Vf[row, col] = random.randint(0, field - 1)
                    else:
                        row = random.randint(0, p * n - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Wf[row, col]
                        Wf[row, col] = random.randint(0, field - 1)

                    # Vectorized verification
                    if verify_vectorized(Uf, Vf, Wf, target_rank, field):
                        return make_solution(Uf, Vf, Wf, target_rank)

                    # Revert if not valid
                    if mat == 'U':
                        Uf[row, col] = old_val
                    elif mat == 'V':
                        Vf[row, col] = old_val
                    else:
                        Wf[row, col] = old_val

    # Return schoolbook (rank 48)
    return make_solution(U, V, W, rank)
# EVOLVE-BLOCK-END
