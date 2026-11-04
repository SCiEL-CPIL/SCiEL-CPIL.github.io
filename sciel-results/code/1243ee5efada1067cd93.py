"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_U = U_schoolbook.tolist()
    best_V = V_schoolbook.tolist()
    best_W = W_schoolbook.tolist()
    best_rank = rank_schoolbook
    
    def is_half_integer(x, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        return np.allclose(2 * x, np.round(2 * x), atol=tol)
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify the tensor decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = np.dot(W[k * n + i, :], U[i * m + j1, :] * V[j2 * p + k, :])
                        expected = 1.0 if j1 == j2 else 0.0
                        if not np.isclose(val, expected, atol=tol):
                            return False
        return True
    
    def solve_for_W(U, V, n, m, p, r):
        """Solve for W given U and V. Returns None if no valid solution."""
        W_solution = np.zeros((p * n, r))
        
        for i in range(n):
            for k in range(p):
                # Build A_ik: (m*m) x r matrix
                # Row (j1,j2): U[i*m+j1, :] * V[j2*p+k, :]
                A_ik = np.zeros((m * m, r))
                b_ik = np.zeros(m * m)
                idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        A_ik[idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        b_ik[idx] = 1.0 if j1 == j2 else 0.0
                        idx += 1
                
                # Solve A_ik * W_ik = b_ik using least squares
                try:
                    W_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=1e-10)
                    W_solution[k * n + i, :] = W_ik
                except np.linalg.LinAlgError:
                    return None
        
        # Verify that the least squares solution is actually exact
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m * m, r))
                b_ik = np.zeros(m * m)
                idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        A_ik[idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        b_ik[idx] = 1.0 if j1 == j2 else 0.0
                        idx += 1
                residual = np.linalg.norm(A_ik @ W_solution[k * n + i, :] - b_ik)
                if residual > 1e-8:
                    return None
        
        return W_solution
    
    def generate_random_UV(n, m, p, r, attempt, rng):
        """Generate diverse random U and V matrices."""
        strategy = attempt % 6
        
        if strategy == 0:
            # Dense random from {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
        elif strategy == 1:
            # Sparse: mostly zeros with some ±1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for _ in range(int(n * m * r * 0.3)):
                U[rng.randint(n * m), rng.randint(r)] = rng.choice([-1, 1])
            for _ in range(int(m * p * r * 0.3)):
                V[rng.randint(m * p), rng.randint(r)] = rng.choice([-1, 1])
        elif strategy == 2:
            # Binary ±1
            U = rng.choice([-1, 1], size=(n * m, r))
            V = rng.choice([-1, 1], size=(m * p, r))
        elif strategy == 3:
            # Half-integer sparse
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for _ in range(int(n * m * r * 0.4)):
                U[rng.randint(n * m), rng.randint(r)] = rng.choice([-0.5, 0.5])
            for _ in range(int(m * p * r * 0.4)):
                V[rng.randint(m * p), rng.randint(r)] = rng.choice([-0.5, 0.5])
        elif strategy == 4:
            # Mixed dense/sparse
            U = rng.choice([-1, -0.5, 0, 0, 0.5, 1], size=(n * m, r))
            V = rng.choice([-1, -0.5, 0, 0, 0.5, 1], size=(m * p, r))
        else:
            # Random signs with varying magnitude
            U = rng.choice([-1, -0.5, 0.5, 1], size=(n * m, r))
            V = rng.choice([-1, -0.5, 0.5, 1], size=(m * p, r))
        
        return U, V
    
    # Try ranks from best_rank-1 down to a reasonable minimum
    # For 3x3x7, theoretical lower bounds suggest rank could be significantly lower
    # Let's try down to at least rank 20
    min_rank_to_try = max(15, best_rank - 45)  # Try ranks 62 down to ~18
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Number of attempts per rank - more for lower ranks
        num_attempts = 50 if r < 30 else 30
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(42 + r * 1000 + attempt)
            U, V = generate_random_UV(n, m, p, r, attempt, rng)
            
            W_solution = solve_for_W(U, V, n, m, p, r)
            if W_solution is None:
                continue
            
            # Check if W has integer/half-integer entries
            if not is_half_integer(W_solution, tol=1e-8):
                continue
            
            # Verify the full decomposition
            if check_decomposition(U, V, W_solution, n, m, p, tol=1e-8):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W_solution.tolist()
                best_rank = r
                break  # Found valid decomposition for this rank, try lower
        
        if time.time() > deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
