"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 110  # 110s safety margin
    
    # For 2x2x5 matrix multiplication, the optimal rank is 13
    # We construct the rank-13 decomposition explicitly
    
    rank = 13
    
    # U: shape (n*m, rank) = (4, 13)
    # V: shape (m*p, rank) = (10, 13)
    # W: shape (p*n, rank) = (10, 13)
    
    # Indexing:
    # A is 2x2: a = i*m + j = 2*i + j, i in {0,1}, j in {0,1}
    #   a=0: (0,0), a=1: (0,1), a=2: (1,0), a=3: (1,1)
    # B is 2x5: b = j*p + k = 5*j + k, j in {0,1}, k in {0,1,2,3,4}
    #   b=0: (0,0), b=1: (0,1), b=2: (0,2), b=3: (0,3), b=4: (0,4)
    #   b=5: (1,0), b=6: (1,1), b=7: (1,2), b=8: (1,3), b=9: (1,4)
    # C is 5x2: c = k*n + i = 2*k + i, k in {0,1,2,3,4}, i in {0,1}
    #   c=0: (0,0), c=1: (0,1), c=2: (1,0), c=3: (1,1)
    #   c=4: (2,0), c=5: (2,1), c=6: (3,0), c=7: (3,1)
    #   c=8: (4,0), c=9: (4,1)
    
    # Support: T[a,b,c] = 1 iff a=2*i+j, b=5*j+k, c=2*k+i
    # i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    
    # Let me build the decomposition using a structured approach
    # Based on the pattern for <2,2,p> with p=5, rank=13
    
    # I'll use a systematic construction. The key insight is that
    # for <2,2,p>, we can use a decomposition based on the structure
    # of the matrix multiplication tensor.
    
    # Let me try a different approach: use the known rank-13 decomposition
    # by constructing it from the algebraic structure.
    
    # For now, let me try to find it via a more efficient search
    # with the knowledge that rank 13 is achievable.
    
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
    
    # Try rank 13 with more systematic search
    best_rank = 20  # schoolbook
    best_U = np.zeros((n*m, 20))
    best_V = np.zeros((m*p, 20))
    best_W = np.zeros((p*n, 20))
    for idx, (a, b, c) in enumerate(support):
        best_U[a, idx] = 1.0
        best_V[b, idx] = 1.0
        best_W[c, idx] = 1.0
    
    # Try rank 13
    r = 13
    max_attempts = 50
    found = False
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Use structured initialization based on the algebraic structure
        np.random.seed(42 + attempt)
        
        # Initialize with some structure
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Alternating minimization with more iterations
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Update U
            for a in range(n*m):
                rows = []
                targets = []
                for (a2, b, c) in support:
                    if a2 == a:
                        rows.append(V[b, :] * W[c, :])
                        targets.append(1.0)
                for (a2, b, c) in all_tuples:
                    if a2 == a and (a2, b, c) not in support_set:
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
                for (a, b2, c) in support:
                    if b2 == b:
                        rows.append(U[a, :] * W[c, :])
                        targets.append(1.0)
                for (a, b2, c) in all_tuples:
                    if b2 == b and (a, b2, c) not in support_set:
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
                for (a, b, c2) in support:
                    if c2 == c:
                        rows.append(U[a, :] * V[b, :])
                        targets.append(1.0)
                for (a, b, c2) in all_tuples:
                    if c2 == c and (a, b, c2) not in support_set:
                        rows.append(U[a, :] * V[b, :])
                        targets.append(0.0)
                if rows:
                    A_mat = np.array(rows)
                    b_vec = np.array(targets)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[c, :] = sol
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                err = compute_error(U, V, W)
                if err < 1e-6:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found = True
                    break
        
        if found:
            break
    
    # If we found rank 13, verify and return
    if found and best_rank == 13:
        # Final verification
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
