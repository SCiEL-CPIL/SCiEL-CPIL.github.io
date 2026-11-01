"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140  # 140s safety margin (10s buffer before 150s kill)
    
    # Build support tensor
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    all_tuples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    support_set = set(support)
    
    def compute_error(U, V, W):
        max_err = 0.0
        for (a, b, c) in support:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        for (a, b, c) in all_tuples:
            if (a, b, c) not in support_set:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Precompute index mappings for efficiency
    # For each a, list of (b, c) in support and non-support
    a_support = {a: [] for a in range(n*m)}
    a_non_support = {a: [] for a in range(n*m)}
    for (a, b, c) in all_tuples:
        if (a, b, c) in support_set:
            a_support[a].append((b, c))
        else:
            a_non_support[a].append((b, c))
    
    b_support = {b: [] for b in range(m*p)}
    b_non_support = {b: [] for b in range(m*p)}
    for (a, b, c) in all_tuples:
        if (a, b, c) in support_set:
            b_support[b].append((a, c))
        else:
            b_non_support[b].append((a, c))
    
    c_support = {c: [] for c in range(p*n)}
    c_non_support = {c: [] for c in range(p*n)}
    for (a, b, c) in all_tuples:
        if (a, b, c) in support_set:
            c_support[c].append((a, b))
        else:
            c_non_support[c].append((a, b))
    
    # Try ranks from 13 upward
    best_rank = 20  # schoolbook fallback
    best_U = np.zeros((n*m, 20))
    best_V = np.zeros((m*p, 20))
    best_W = np.zeros((p*n, 20))
    for idx, (a, b, c) in enumerate(support):
        best_U[a, idx] = 1.0
        best_V[b, idx] = 1.0
        best_W[c, idx] = 1.0
    
    found = False
    
    for r in [13, 14, 15]:
        if time.time() > deadline or found:
            break
        
        # Multiple random restarts
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + r * 100 + attempt)
            
            # Initialize with structured approach: use Strassen-like structure for first part
            # For 2x2x2, Strassen gives rank 7. For 2x2x5, we can use Strassen for first 3 columns
            # and schoolbook for last 2, giving 7 + 2*2 = 11... but that's not quite right.
            
            # Let's use a smarter initialization: start with schoolbook-like structure
            # and add some randomness
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            # Add schoolbook structure as a hint
            for idx, (a, b, c) in enumerate(support):
                if idx < r:
                    U[a, idx] += 1.0
                    V[b, idx] += 1.0
                    W[c, idx] += 1.0
            
            # Alternating least squares
            for iteration in range(300):
                if time.time() > deadline:
                    break
                
                # Update U using precomputed indices
                for a in range(n*m):
                    rows = []
                    targets = []
                    for (b, c) in a_support[a]:
                        rows.append(V[b, :] * W[c, :])
                        targets.append(1.0)
                    for (b, c) in a_non_support[a]:
                        rows.append(V[b, :] * W[c, :])
                        targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[a, :] = sol
                
                # Update V
                for b in range(m*p):
                    rows = []
                    targets = []
                    for (a, c) in b_support[b]:
                        rows.append(U[a, :] * W[c, :])
                        targets.append(1.0)
                    for (a, c) in b_non_support[b]:
                        rows.append(U[a, :] * W[c, :])
                        targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[b, :] = sol
                
                # Update W
                for c in range(p*n):
                    rows = []
                    targets = []
                    for (a, b) in c_support[c]:
                        rows.append(U[a, :] * V[b, :])
                        targets.append(1.0)
                    for (a, b) in c_non_support[c]:
                        rows.append(U[a, :] * V[b, :])
                        targets.append(0.0)
                    if rows:
                        A_mat = np.array(rows)
                        b_vec = np.array(targets)
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[c, :] = sol
                
                # Check convergence every 5 iterations
                if iteration % 5 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-7:
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    # If we found a valid decomposition, verify and return
    if found:
        final_err = compute_error(best_U, best_V, best_W)
        if final_err < 1e-5:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
    
    # Fallback: return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
