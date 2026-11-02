import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Precompute active triples
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
                
    # For each c, precompute the active (a,b) pairs
    # c = k*n + i
    active_ab_for_c = {c: [] for c in range(p * n)}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_ab_for_c[c].append((a, b))
                
    # All (a,b) pairs
    all_ab = [(a, b) for a in range(n * m) for b in range(m * p)]
    all_ab_set = set(all_ab)
    
    def check_decomposition(U, V, W, R):
        """Verify U, V, W with rank R."""
        # U, V, W are numpy arrays
        for a, b, c in active_triples:
            s = np.dot(U[a], V[b]) # No, this is wrong.
            # Sum over r: U[a,r] * V[b,r] * W[c,r]
            # We can compute this as element-wise product then sum
            s = np.sum(U[a] * V[b] * W[c])
            if s != 1:
                return False
        for a, b, c in itertools.product(range(n * m), range(m * p), range(p * n)):
            if (a, b, c) in active_triples:
                continue
            s = np.sum(U[a] * V[b] * W[c])
            if s != 0:
                return False
        return True

    def solve_for_W(U, V, R):
        """
        Given U and V, try to solve for W such that the tensor equations hold.
        Returns W if an integer solution exists, else None.
        """
        W = np.zeros((p * n, R), dtype=float)
        for c in range(p * n):
            # Construct A_c (108 x R) and b_c (108)
            A_c = np.zeros((len(all_ab), R))
            b_c = np.zeros(len(all_ab))
            for idx, (a, b) in enumerate(all_ab):
                # A_c[idx, r] = U[a, r] * V[b, r]
                A_c[idx] = U[a] * V[b]
                if (a, b, c) in active_triples:
                    b_c[idx] = 1
                else:
                    b_c[idx] = 0
            
            # Solve A_c w = b_c
            # Use least squares to check consistency
            # Since R is small, we can use np.linalg.lstsq
            try:
                w_c, residuals, rank, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            # Check if residual is close to 0
            if len(residuals) > 0 and residuals[0] > 1e-9:
                return None
            
            # Check if w_c is integer
            w_c_int = np.rint(w_c)
            if not np.allclose(w_c, w_c_int, atol=1e-9):
                return None
            
            W[c] = w_c_int
            
        return W.astype(int)

    def random_init_UV(R, scale=2):
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        for i in range(n * m):
            for r in range(R):
                if random.random() < 0.5:
                    U[i, r] = random.randint(-scale, scale)
        for i in range(m * p):
            for r in range(R):
                if random.random() < 0.5:
                    V[i, r] = random.randint(-scale, scale)
        return U, V

    def local_search_UV(U, V, R, max_iters=1000):
        """
        Local search on U and V to find a configuration where solve_for_W succeeds.
        We use a penalty function based on the residuals of the linear systems.
        """
        current_U = U.copy()
        current_V = V.copy()
        
        def penalty(U, V):
            total_pen = 0.0
            for c in range(p * n):
                A_c = np.zeros((len(all_ab), R))
                b_c = np.zeros(len(all_ab))
                for idx, (a, b) in enumerate(all_ab):
                    A_c[idx] = U[a] * V[b]
                    if (a, b, c) in active_triples:
                        b_c[idx] = 1
                try:
                    w_c, residuals, rank, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                    if len(residuals) > 0:
                        total_pen += residuals[0]
                    # Add penalty for non-integer solution
                    w_c_int = np.rint(w_c)
                    total_pen += np.sum((w_c - w_c_int) ** 2)
                except:
                    total_pen += 1000.0
            return total_pen
        
        current_pen = penalty(current_U, current_V)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random entry in U or V to modify
            if random.random() < 0.5:
                idx = random.randint(0, n * m - 1)
                r = random.randint(0, R - 1)
                old_val = current_U[idx, r]
                current_U[idx, r] = old_val + random.choice([-1, 1])
                new_pen = penalty(current_U, current_V)
                if new_pen < current_pen:
                    current_pen = new_pen
                else:
                    current_U[idx, r] = old_val
            else:
                idx = random.randint(0, m * p - 1)
                r = random.randint(0, R - 1)
                old_val = current_V[idx, r]
                current_V[idx, r] = old_val + random.choice([-1, 1])
                new_pen = penalty(current_U, current_V)
                if new_pen < current_pen:
                    current_pen = new_pen
                else:
                    current_V[idx, r] = old_val
            
            # Check for success
            if current_pen < 1e-9:
                W = solve_for_W(current_U, current_V, R)
                if W is not None:
                    return current_U, current_V, W
            
            if it % 100 == 0:
                W = solve_for_W(current_U, current_V, R)
                if W is not None:
                    return current_U, current_V, W
        return None

    # Search for lower ranks
    # Start from 15
    for R in range(15, best_rank):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 if R < 20 else 20
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            U, V = random_init_UV(R, scale=2)
            res = local_search_UV(U, V, R, max_iters=500)
            if res is not None:
                U_sol, V_sol, W_sol = res
                # Final verification
                if check_decomposition(U_sol, V_sol, W_sol, R):
                    best_result = {
                        "U": U_sol.tolist(),
                        "V": V_sol.tolist(),
                        "W": W_sol.tolist()
                    }
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
            
    return best_result
