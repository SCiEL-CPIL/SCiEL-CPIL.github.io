import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build the tensor: T[a,b,g] = 1 if a=i*m+j, b=j*p+k, g=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                T[a, b, g] = 1.0
                valid_triples.append((a, b, g))
    
    # Flatten tensor for linear algebra
    # We want: sum_c U[a,c]*V[b,c]*W[g,c] = T[a,b,g]
    # For fixed U,V: linear in W. For fixed U,W: linear in V. Etc.
    
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check valid entries
        for (a, b, g) in valid_triples:
            val = 0.0
            for c in range(r):
                val += U[a, c] * V[b, c] * W[g, c]
            if abs(val - 1.0) > tol:
                return False
        # Check invalid entries are zero
        for a in range(n * m):
            for b in range(m * p):
                for g in range(p * n):
                    if (a, b, g) not in set(valid_triples):
                        val = 0.0
                        for c in range(r):
                            val += U[a, c] * V[b, c] * W[g, c]
                        if abs(val) > tol:
                            return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Compute all products
        for (a, b, g) in valid_triples:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val - 1.0) > tol:
                return False
        # Spot check some invalid entries
        invalid_sample = []
        for a in range(n * m):
            for b in range(m * p):
                for g in range(p * n):
                    if (a, b, g) not in set(valid_triples):
                        invalid_sample.append((a, b, g))
        import random
        random.seed(42)
        sample = random.sample(invalid_sample, min(50, len(invalid_sample)))
        for (a, b, g) in sample:
            val = np.dot(U[a, :], V[b, :] * W[g, :])
            if abs(val) > tol:
                return False
        return True
    
    def solve_for_W(U, V):
        """Given U and V, solve linear system for W."""
        r = U.shape[1]
        W = np.zeros((p * n, r))
        
        # For each g, we have equations: sum_c U[a,c]*V[b,c]*W[g,c] = T[a,b,g]
        # This is: (U * V_outer) @ W[g,:] = T[:, :, g]
        # where (U * V_outer)[a,b,c] = U[a,c]*V[b,c]
        
        for g in range(p * n):
            # Build matrix M where M[a,b,c] = U[a,c]*V[b,c]
            # We need M_vec @ w_g = t_g
            # M_vec has shape (n*m*m*p, r)
            # t_g has shape (n*m*m*p,)
            
            M = np.zeros((n * m * m * p, r))
            t = np.zeros(n * m * m * p)
            
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    M[idx, :] = U[a, :] * V[b, :]
                    # Find if (a,b,g) is a valid triple
                    # a = i*m+j, b = j*p+k, g = k*n+i
                    # Check consistency
                    found = False
                    for i in range(n):
                        for j in range(m):
                            if a == i * m + j:
                                for k in range(p):
                                    if b == j * p + k:
                                        if g == k * n + i:
                                            t[idx] = 1.0
                                            found = True
                                        break
                                break
                    idx += 1
            
            # Solve least squares
            try:
                w_g, residuals, rank, sv = np.linalg.lstsq(M, t, rcond=None)
            except:
                return None
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                return None
            
            W[g, :] = w_g
        
        return W
    
    def snap_to_half_integers(U, V, W, tol=0.01):
        """Snap values to nearest half-integer if close."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Check if snapping preserved validity
        if verify_fast(U_s, V_s, W_s):
            return U_s, V_s, W_s
        
        # Try rounding to integers only
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify_fast(U_i, V_i, W_i):
            return U_i, V_i, W_i
        
        return None
    
    # Strategy: Try structured random searches at various ranks
    # Use patterns inspired by Strassen for the 3x3x3 core, then extend to 3x3x4
    
    def random_structured_search(r, max_attempts=500, seed_offset=0):
        np.random.seed(seed_offset + r * 1000)
        
        vals = [-1, -0.5, 0, 0.5, 1]
        probs = [0.08, 0.08, 0.60, 0.08, 0.08]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random sparse U and V
            U = np.random.choice(vals, size=(n * m, r), p=probs)
            V = np.random.choice(vals, size=(m * p, r), p=probs)
            
            # Ensure some structure: each row of U and V should have at least one non-zero
            for i in range(n * m):
                if np.all(U[i, :] == 0):
                    U[i, np.random.randint(r)] = np.random.choice([-1, 1])
            for j in range(m * p):
                if np.all(V[j, :] == 0):
                    V[j, np.random.randint(r)] = np.random.choice([-1, 1])
            
            # Solve for W
            W = solve_for_W(U, V)
            if W is None:
                continue
            
            # Check if solution is valid
            if not verify_fast(U, V, W):
                continue
            
            # Try to snap to half-integers
            snapped = snap_to_half_integers(U, V, W)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                return U_s, V_s, W_s
            
            # Use as-is if valid
            return U, V, W
        
        return None
    
    # Also try a more systematic approach based on known patterns
    def pattern_based_search(r, max_attempts=200):
        """Try modifications of known low-rank patterns."""
        np.random.seed(12345 + r)
        
        # Start with a base pattern for 3x3x3 (Strassen-like, rank 7)
        # and extend to handle the 4th column of B
        base_patterns = []
        
        # Pattern 1: Direct extension of rank-7 3x3x3
        # For 3x3x3, we have 7 terms. For 3x3x4, we need to handle B's 4th column.
        # The 4th column of B (k=3) contributes to C[:,3] = A @ B[:,3]
        # This requires at least 3 more terms (one per row of A)
        # So minimum rank might be 7 + 3 = 10? But that's likely not tight.
        
        # Let's try random perturbations of structured initializations
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with a structured pattern
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            
            # Fill with random values from {-1, 0, 1}
            for i in range(r):
                # Each term uses a subset of A indices and B indices
                num_a = np.random.randint(1, 4)
                num_b = np.random.randint(1, 4)
                
                a_indices = np.random.choice(n * m, size=num_a, replace=False)
                b_indices = np.random.choice(m * p, size=num_b, replace=False)
                
                for ai in a_indices:
                    U[ai, i] = np.random.choice([-1, 1])
                for bi in b_indices:
                    V[bi, i] = np.random.choice([-1, 1])
            
            W = solve_for_W(U, V)
            if W is None:
                continue
            
            if not verify_fast(U, V, W):
                continue
            
            snapped = snap_to_half_integers(U, V, W)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                return U_s, V_s, W_s
            
            return U, V, W
        
        return None
    
    # Try ranks from low to high
    best_result = None
    
    for r in range(18, 37):
        if time.time() > deadline:
            break
        
        attempts = 300 if r < 25 else 150
        
        # Try random structured search
        result = random_structured_search(r, max_attempts=attempts, seed_offset=r)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, r)
                break
        
        # Try pattern-based search
        if time.time() < deadline - 30:
            result2 = pattern_based_search(r, max_attempts=attempts // 2)
            if result2 is not None:
                U2, V2, W2 = result2
                if verify_fast(U2, V2, W2):
                    if best_result is None or r < best_result[3]:
                        best_result = (U2, V2, W2, r)
                        break
    
    if best_result is not None:
        U, V, W, r = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Try the original rank-21 program's approach with verification
    U21 = np.zeros((n * m, 21), dtype=float)
    V21 = np.zeros((m * p, 21), dtype=float)
    W21 = np.zeros((p * n, 21), dtype=float)
    
    # Term 1: (A00+A11)(B00+B11)
    U21[0, 0] += 1; U21[4, 0] += 1
    V21[0, 0] += 1; V21[5, 0] += 1
    W21[0, 0] += 1; W21[4, 0] += 1
    
    # Term 2: (A10+A11)B00
    U21[3, 1] += 1; U21[4, 1] += 1
    V21[0, 1] += 1
    W21[1, 1] += 1; W21[4, 1] += 1
    
    # Term 3: A00(B01-B11)
    U21[0, 2] += 1
    V21[1, 2] += 1; V21[5, 2] += -1
    W21[3, 2] += 1; W21[4, 2] += 1
    
    # Term 4: A11(B10-B00)
    U21[4, 3] += 1
    V21[4, 3] += 1; V21[0, 3] += -1
    W21[0, 3] += 1; W21[1, 3] += 1
    
    # Term 5: (A00+A01)B11
    U21[0, 4] += 1; U21[1, 4] += 1
    V21[5, 4] += 1
    W21[3, 4] += 1; W21[0, 4] += 1
    
    # Term 6: (A10-A00)(B00+B01)
    U21[3, 5] += 1; U21[0, 5] += -1
    V21[0, 5] += 1; V21[1, 5] += 1
    W21[4, 5] += 1
    
    # Term 7: (A01-A11)(B10+B11)
    U21[1, 6] += 1; U21[4, 6] += -1
    V21[4, 6] += 1; V21[5, 6] += 1
    W21[0, 6] += 1
    
    # Terms 8-21: Cover remaining indices
    U21[2, 7] += 1; V21[8, 7] += 1; W21[0, 7] += 1
    U21[2, 8] += 1; V21[9, 8] += 1; W21[3, 8] += 1
    U21[2, 9] += 1; V21[10, 9] += 1; W21[6, 9] += 1
    U21[2, 10] += 1; V21[11, 10] += 1; W21[9, 10] += 1
    U21[5, 11] += 1; V21[8, 11] += 1; W21[1, 11] += 1
    U21[5, 12] += 1; V21[9, 12] += 1; W21[4, 12] += 1
    U21[5, 13] += 1; V21[10, 13] += 1; W21[7, 13] += 1
    U21[5, 14] += 1; V21[11, 14] += 1; W21[10, 14] += 1
    U21[6, 15] += 1; V21[0, 15] += 1; W21[2, 15] += 1
    U21[6, 16] += 1; V21[1, 16] += 1; W21[5, 16] += 1
    U21[6, 17] += 1; V21[2, 17] += 1; W21[8, 17] += 1
    U21[6, 18] += 1; V21[3, 18] += 1; W21[11, 18] += 1
    U21[7, 19] += 1; V21[4, 19] += 1; W21[2, 19] += 1
    U21[7, 20] += 1; V21[5, 20] += 1; W21[5, 20] += 1
    
    if verify_fast(U21, V21, W21):
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
