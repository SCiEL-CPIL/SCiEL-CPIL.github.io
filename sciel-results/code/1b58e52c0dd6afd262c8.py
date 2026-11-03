import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    schoolbook_rank = n * m * p  # 144
    
    # Precompute target: for each (u,v), what w should have T=1?
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
    
    # Precompute all (u,v) pairs and their target w
    uv_pairs = []
    for u in range(nm):
        for v in range(mp):
            uv_pairs.append((u, v, target_w[(u, v)]))
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = schoolbook_rank
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
    deadline = start_time + 145  # 5s safety margin
    
    best_solution = schoolbook()
    best_rank = schoolbook_rank
    
    def verify_solution(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        # Check integer/half-integer entries
        for mat in [U, V, W]:
            scaled = mat * 2
            if not np.allclose(scaled, np.round(scaled), atol=1e-9):
                return False
        
        # Check tensor equation
        for u, v, tw in uv_pairs:
            for w in range(pn):
                expected = 1.0 if (tw == w and tw >= 0) else 0.0
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - expected) > 1e-9:
                    return False
        return True
    
    def try_rank(r, num_trials):
        """Try to find a valid decomposition of rank r."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random U and V with entries from {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, (nm, r))
            V = np.random.randint(-2, 3, (mp, r))
            
            # For each w, solve for W[w, :]
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                # Build A (nm*mp x r) and b (nm*mp,)
                A = np.zeros((nm * mp, r))
                b = np.zeros(nm * mp)
                idx = 0
                for u, v, tw in uv_pairs:
                    A[idx, :] = U[u, :] * V[v, :]
                    b[idx] = 1.0 if (tw == w) else 0.0
                    idx += 1
                
                # Solve A @ W_w = b
                try:
                    W_w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    valid = False
                    break
                
                W[w, :] = W_w
            
            if not valid:
                continue
            
            # Check if W is close to half-integers
            W_scaled = W * 2
            W_rounded = np.round(W_scaled)
            if not np.allclose(W_scaled, W_rounded, atol=1e-8):
                continue
            
            W_final = W_rounded / 2.0
            
            # Verify
            if verify_solution(U, V, W_final):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_final.tolist()
                }
        
        return None
    
    # Try ranks from low to high
    # Lower bound: at least max(nm, mp, pn) = 36, but realistically higher
    # Try from 40 up to 143
    for r in range(40, best_rank):
        if time.time() > deadline - 3:
            break
        
        time_left = deadline - time.time()
        if time_left < 1:
            break
        
        # Allocate trials based on time
        num_trials = max(10, int(time_left * 15))
        
        result = try_rank(r, num_trials)
        if result is not None:
            best_solution = result
            best_rank = r
            break
    
    return best_solution
