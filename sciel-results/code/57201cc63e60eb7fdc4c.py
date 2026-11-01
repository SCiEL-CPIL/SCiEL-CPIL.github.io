import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R
    
    best_U, best_V, best_W, best_R = schoolbook()
    
    def verify(U, V, W, R):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(R, max_attempts=20, time_budget=15.0):
        """Try to find a valid decomposition of rank R using random search + local fix."""
        start = time.time()
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for attempt in range(max_attempts):
            if time.time() > start + time_budget or time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(attempt * 7919 + R * 104729)
            
            # Initialize with sparse random values
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            W = np.zeros((p * n, R), dtype=float)
            
            # Fill with random small values, biased toward sparsity
            for idx in range(n * m):
                for r in range(R):
                    if rng.random() < 0.3:
                        U[idx, r] = rng.choice(vals)
            for idx in range(m * p):
                for r in range(R):
                    if rng.random() < 0.3:
                        V[idx, r] = rng.choice(vals)
            for idx in range(p * n):
                for r in range(R):
                    if rng.random() < 0.3:
                        W[idx, r] = rng.choice(vals)
            
            # Local search: iteratively fix violations
            for iteration in range(5000):
                if time.time() > start + time_budget or time.time() > deadline - 5:
                    break
                
                # Find violations
                violations = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            if abs(s - 1) > 1e-9:
                                violations.append((i, j, k, s))
                
                if not violations:
                    # Valid! Convert to integer/half-integer representation
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    # Verify with rounded values
                    if verify(U_int, V_int, W_int, R):
                        return U_int, V_int, W_int, R
                    else:
                        # Try with original values
                        return U, V, W, R
                
                # Pick a random violation and try to fix it
                vi, vj, vk, vs = rng.choice(violations) if len(violations) > 1 else violations[0]
                target_diff = 1 - vs
                
                # Try adjusting one of the entries in a random column
                r = rng.integers(0, R)
                ui = U[vi * m + vj, r]
                vj_idx = vj * p + vk
                vk_idx = vk * n + vi
                vval = V[vj_idx, r]
                wval = W[vk_idx, r]
                
                # Try different adjustments
                for new_u in vals:
                    for new_v in vals:
                        for new_w in vals:
                            # Compute new sum
                            old_contrib = ui * vval * wval
                            new_contrib = new_u * new_v * new_w
                            if abs(new_contrib - old_contrib) > 1e-12:
                                # Check if this helps
                                if abs(target_diff + (new_contrib - old_contrib)) < abs(target_diff) * 0.5 + 0.01:
                                    U[vi * m + vj, r] = new_u
                                    V[vj_idx, r] = new_v
                                    W[vk_idx, r] = new_w
                                    break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
        
        return None
    
    # Try ranks below schoolbook
    for R in range(best_R - 1, max(10, best_R - 15), -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(R, max_attempts=15, time_budget=8.0)
        if result is not None:
            best_U, best_V, best_W, best_R = result
            # Keep trying for even lower rank
            continue
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback to schoolbook
        best_U, best_V, best_W, best_R = schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
