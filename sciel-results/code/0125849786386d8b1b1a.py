"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = nm * mp * pn // (m * m)  # n*m*p = 144
    
    # Precompute the tensor target: for each (u,v), what w should have T=1?
    # u = i*m + j, v = j'*p + k, w = k'*n + i'
    # T[u,v,w] = 1 iff j == j' AND i == i' (where i from u, j from u, j' from v, k from v, k' from w, i' from w)
    # Actually: T[u,v,w] = 1 iff the indices are consistent for matrix mult
    # u = i*m+j means row i of A, col j of A / row j of B
    # v = j'*p+k means row j' of B, col k of B / col k of C
    # w = k'*n+i' means row k' of C, col i' of C
    # Consistency: j == j' (A's col matches B's row) AND i == i' (A's row matches C's col) AND k == k' (B's col matches C's row)
    # So for given (u,v): i = u//m, j = u%m, j' = v//p, k = v%p
    # If j != j': no valid w exists (T is all zeros for this (u,v))
    # If j == j': we need w such that k' == k and i' == i, so w = k*n + i
    
    target_w = {}
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            jp = v // p
            k = v % p
            if j == jp:
                target_w[(u, v)] = k * n + i
            else:
                target_w[(u, v)] = -1
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_solution = schoolbook()
    best_rank = n * m * p
    
    def is_valid_solution(U, V, W):
        """Check if U, V, W (numpy arrays) form a valid decomposition."""
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            scaled = mat * 2
            if not np.allclose(scaled, np.round(scaled)):
                return False
            if not np.all(np.abs(scaled - np.round(scaled)) < 1e-9):
                return False
        # Check the tensor equation
        # For each (u,v,w): sum_c U[u,c]*V[v,c]*W[w,c] should equal T[u,v,w]
        # This is expensive to check fully, so we'll check a sample
        # Actually, let's check all since dimensions are small
        for u in range(nm):
            for v in range(mp):
                tw = target_w[(u, v)]
                for w in range(pn):
                    expected = 1.0 if w == tw else 0.0
                    # sum_c U[u,c] * V[v,c] * W[w,c]
                    val = np.dot(U[u, :] * V[v, :], W[w, :])
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(r, num_trials=50):
        """Try to find a valid decomposition of rank r."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random U and V with small integer entries
            # Use entries from {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, (nm, r))
            V = np.random.randint(-2, 3, (mp, r))
            
            # For each w, solve for W[w, :]
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                # Build system: for all (u,v), sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
                # Let X[u,v] = U[u,:] * V[v,:] (element-wise, shape (r,))
                # Then we need: X[u,v] @ W[w,:] = T[u,v,w]
                # This is a linear system: A @ W[w,:] = b
                # where A[u,v,:] = X[u,v] and b[u,v] = T[u,v,w]
                
                # Build A (nm*mp x r) and b (nm*mp,)
                A = np.zeros((nm * mp, r))
                b = np.zeros(nm * mp)
                idx = 0
                for u in range(nm):
                    for v in range(mp):
                        A[idx, :] = U[u, :] * V[v, :]
                        tw = target_w[(u, v)]
                        b[idx] = 1.0 if (tw == w) else 0.0
                        idx += 1
                
                # Solve A @ W_w = b using least squares
                # Use np.linalg.lstsq
                try:
                    W_w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    valid = False
                    break
                
                W[w, :] = W_w
            
            if not valid:
                continue
            
            # Check if W has integer/half-integer entries
            W_scaled = W * 2
            W_rounded = np.round(W_scaled)
            if not np.allclose(W_scaled, W_rounded, atol=1e-9):
                continue
            
            # Convert to actual values
            W_final = W_rounded / 2.0
            
            # Verify the full solution
            if is_valid_solution(U, V, W_final):
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_final.tolist()
                }
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(nm, mp, pn) = max(16, 36, 36) = 36
    # But realistically, for matrix mult, rank is at least ~n*p or similar
    # Let's try from 40 up to 143
    for r in range(40, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Number of trials based on time remaining
        time_left = deadline - time.time()
        if time_left < 2:
            break
        num_trials = min(100, int(time_left * 20))
        
        result = try_rank(r, num_trials)
        if result is not None:
            best_solution = result
            best_rank = r
            # Don't break - try to find even lower? No, we're going up.
            # But we found one at rank r, so we can stop since we want minimal
            break
    
    return best_solution
# EVOLVE-BLOCK-END
