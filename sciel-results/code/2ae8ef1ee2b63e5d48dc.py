import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin
    
    n, m, p = 3, 3, 4
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Target rank range
    # Schoolbook is 36.
    # Try to find lower rank.
    
    def verify(U, V, W, r):
        # U: (N, r), V: (M, r), W: (P, r)
        # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for rr in range(r):
                        s += U[u_idx, rr] * V[v_idx, rr] * W[w_idx, rr]
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def solve_block(A, target):
        # A is (m, r), target is (m, 1)
        # Solve A x = target
        # Returns x if solution exists and is "nice", else None
        # We want x to be half-integers
        try:
            # Find a particular solution
            # Use least squares if not consistent, but we need exact consistency first
            # Check consistency
            rank_A = np.linalg.matrix_rank(A)
            rank_aug = np.linalg.matrix_rank(np.hstack([A, target.reshape(-1, 1)]))
            if rank_A != rank_aug:
                return None
            
            # Get particular solution
            # If r == m, use inv. If r > m, use lstsq (gives min norm)
            if r == m:
                x = np.linalg.solve(A, target)
            else:
                x = np.linalg.lstsq(A, target, rcond=None)[0]
            
            # Check if x is close to half-integers
            # Scale by 2
            x_scaled = x * 2
            x_rounded = np.round(x_scaled)
            if np.allclose(x_scaled, x_rounded, atol=1e-7):
                return x_rounded / 2.0
            else:
                # Try to find a solution in the null space that is half-integer?
                # This is hard. Just return None for now.
                return None
        except np.linalg.LinAlgError:
            return None

    # Try ranks from 20 to 35
    for r in range(20, 36):
        if time.time() > DEADLINE:
            break
            
        attempts = 500
        for attempt in range(attempts):
            if time.time() > DEADLINE:
                break
                
            # Generate random U, V
            # Use sparse random matrices with values -1, 0, 1
            # Density 0.5
            U = np.random.choice([-1, 0, 1], size=(N, r))
            V = np.random.choice([-1, 0, 1], size=(M, r))
            
            W = np.zeros((P, r))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Construct A for this block
                    # A[j, rr] = U[i*m+j, rr] * V[j*p+k, rr] for j in 0..m-1
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A[j, :] = U[u_idx, :] * V[v_idx, :]
                    
                    target = np.ones(m)
                    x = solve_block(A, target)
                    if x is None:
                        valid = False
                        break
                    W[w_idx, :] = x
                    if time.time() > DEADLINE:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the full solution
                if verify(U, V, W, r):
                    # Convert to lists for output
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback: Schoolbook
    r = 36
    U = np.zeros((N, r))
    V = np.zeros((M, r))
    W = np.zeros((P, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
