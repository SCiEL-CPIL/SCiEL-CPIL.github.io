# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    # Build the 28-term Strassen basis for the 2x2x8 part
    rank_base = 28
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U_base[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V_base[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W_base[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Try to absorb the a=2 part into these 28 terms
    # We need to find U_ext (2 x 28) and V_ext (8 x 28) such that:
    # sum_r U_ext[i, r] * V_ext[k, r] * W_base[k*n+i, r] == 1 for all i, k
    
    # Strategy: Fix U_ext[i, r] = 1. Solve for V_ext.
    # Use exact rational arithmetic to check for half-integer solutions.
    
    U_ext = [[1] * rank_base for _ in range(n)]
    V_ext = [[0] * rank_base for _ in range(p)]
    
    success = True
    for k in range(p):
        # Build system for this k
        # A_k is n x rank_base
        A_k = []
        for i in range(n):
            row_idx = k * n + i
            A_k.append(W_base[row_idx][:])
        
        b_k = [1, 1]
        
        # Solve A_k * v = b_k
        # A_k is 2 x 28.
        # We want v with half-integer entries.
        
        # Use numpy to get a particular solution, then check/adjust?
        # Or use exact linear algebra.
        
        # Since the system is underdetermined, we can pick a solution.
        # Let's try to find a solution with minimal support or small values.
        
        # Convert to Fractions
        A_frac = [[Fraction(x) for x in row] for row in A_k]
        b_frac = [Fraction(x) for x in b_k]
        
        # Perform Gaussian elimination with Fractions to find a solution
        # We have 2 equations, 28 unknowns.
        # We can set 26 variables to 0 and solve for the 2?
        # Or set 2 variables to 0 and solve for the rest?
        # To get half-integers, we need the solution to be in (1/2)Z^28.
        
        # Let's try to solve for the first 2 variables (v[0], v[1]) assuming others are 0?
        # This might not work if the submatrix is singular.
        
        # Better: Find the null space basis?
        # Or just use `np.linalg.lstsq` and check if rounding works.
        # The previous code did this. Why did it fail?
        # Maybe the solution wasn't close to half-integers.
        
        # Let's try a different U_ext.
        # If we vary U_ext, the system for V_ext changes.
        # We can try a few random U_ext vectors?
        
        # For now, stick to U_ext = 1 and use exact solve.
        
        # Simple Gaussian Elimination for 2x28
        # We can express v[2] ... v[27] in terms of v[0], v[1]?
        # Or v[0], v[1] in terms of others?
        
        # Let's try to find a solution where v[2..27] = 0.
        # Solve [[A_k[0,0], A_k[0,1]], [A_k[1,0], A_k[1,1]]] * [v0, v1] = b
        # If this 2x2 matrix is invertible, we have a solution.
        
        det = A_frac[0][0] * A_frac[1][1] - A_frac[0][1] * A_frac[1][0]
        if det != 0:
            v0 = (b_frac[0] * A_frac[1][1] - A_frac[0][1] * b_frac[1]) / det
            v1 = (A_frac[0][0] * b_frac[1] - b_frac[0] * A_frac[1][0]) / det
            
            # Check if v0, v1 are half-integers
            if (v0 * 2).denominator == 1 and (v1 * 2).denominator == 1:
                v_k = [0] * rank_base
                v_k[0] = v0
                v_k[1] = v1
                V_ext[k] = [float(x) for x in v_k] # Store as float for now, will convert to fraction later?
                # Actually, we need to store the exact values.
                # Let's store as Fraction in a separate list?
                # The final output needs to be numbers.
                # We can store as float if they are exact half-integers.
                continue
            else:
                # Try other pairs of variables
                pass
        
        # If the simple 2-var solve fails, try more variables.
        # For simplicity, if the first attempt fails, mark as failed.
        # We can expand this to try all pairs (i, j).
        
        found = False
        for i1 in range(rank_base):
            if found: break
            for i2 in range(rank_base):
                if i1 == i2: continue
                # Solve for v[i1], v[i2]
                a11 = A_frac[0][i1]
                a12 = A_frac[0][i2]
                a21 = A_frac[1][i1]
                a22 = A_frac[1][i2]
                
                det = a11 * a22 - a12 * a21
                if det == 0: continue
                
                v_i1 = (b_frac[0] * a22 - a12 * b_frac[1]) / det
                v_i2 = (a11 * b_frac[1] - b_frac[0] * a21) / det
                
                if (v_i1 * 2).denominator == 1 and (v_i2 * 2).denominator == 1:
                    v_k = [0] * rank_base
                    v_k[i1] = v_i1
                    v_k[i2] = v_i2
                    V_ext[k] = [float(x) for x in v_k]
                    found = True
                    break
        
        if not found:
            success = False
            break

    if success:
        # Construct the final candidate with rank 28
        U_final = [row[:] for row in U_base]
        V_final = [row[:] for row in V_base]
        W_final = [row[:] for row in W_base]
        
        # Add U_ext to U_final (indices for a=2)
        for i in range(n):
            for r in range(rank_base):
                U_final[i * m + 2][r] = U_ext[i][r]
                
        # Add V_ext to V_final (indices for a=2)
        for k in range(p):
            for r in range(rank_base):
                V_final[2 * p + k][r] = V_ext[k][r]
                
        candidate = {"U": U_final, "V": V_final, "W": W_final}
        
        if _validate(candidate):
            return candidate
        else:
            return _schoolbook()
    else:
        # Fallback to 44
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Copy base
        for r in range(rank_base):
            for i in range(n):
                for a in range(2):
                    U[i*m+a][r] = U_base[i*m+a][r]
            for a in range(2):
                for k in range(p):
                    V[a*p+k][r] = V_base[a*p+k][r]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = W_base[k*n+i][r]
                    
        col = rank_base
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if not _elapsed_ok():
                    return _schoolbook()

        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate
        else:
            return _schoolbook()

# EVOLVE-BLOCK-END
