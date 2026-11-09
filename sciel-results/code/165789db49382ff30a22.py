"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S = [
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        ({(2,1),(3,1)}, {(0,1)}, {(1,1),(3,-1)}),
        ({(0,1)}, {(1,1),(3,-1)}, {(2,1),(3,1)}),
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(1,1)}),
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(2,1)}),
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),
    ]
    
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=4, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_find_rank12_sub():
    """
    Try to find a rank 12 decomposition for M_{2,4,2} with half-integer entries.
    Uses a structured approach based on the algebraic structure of the problem.
    M_{2,4,2}: (2x4) * (4x2) -> (2x2). Each output (i,k) = sum_j A[i,j]*B[j,k].
    We need 12 rank-1 terms. With half-integer entries (multiply by 2 for integers).
    """
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 12
    
    # The key structure: we have 4 output equations (i,k) for i in {0,1}, k in {0,1}
    # Each equation: sum_j U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r = 1 (or 2 for integer version)
    
    # Strategy: Use a "block" structure where we split the 4 j-indices into pairs
    # and use Strassen-like additions/subtractions
    
    # Approach: Try all assignments of small integers to a structured template
    # Template: 12 columns = 6 pairs, each pair handles a "Strassen term"
    
    # Let's use a more direct approach: enumerate sparse patterns
    # For each column r, U has 8 entries (2*4), V has 8 entries (4*2)
    # We want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta (for each i,j,k)
    
    # Actually the condition is: for each (i,k), sum_{j,r} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # And for each (i,j,k) the "tensor" T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1 if we're computing the dot product correctly
    
    # Wait - the tensor for matrix multiplication T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # No - T[i,j,k] is the coefficient of A[i,j]*B[j,k] in the output. For standard matmul,
    # output[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid (i,j,k).
    
    # So we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for ALL i in {0,1}, j in {0,1,2,3}, k in {0,1}
    
    # That's 2*4*2 = 16 equations.
    
    # With integer entries (multiply by 2): sum_r U'[i,j,r]*V'[j,k,r]*W'[k,i,r] = 2
    # where U' = 2*U, etc.
    
    # Let's try a structured approach with specific patterns
    
    # Pattern: Use 12 columns where columns come in 3 groups of 4
    # Group 1 (cols 0-3): handle j=0,1 contributions
    # Group 2 (cols 4-7): handle j=2,3 contributions  
    # Group 3 (cols 8-11): cross terms
    
    # Actually, let me try a simpler structured search:
    # Fix V to have a simple pattern and solve for U, W
    
    def solve_for_UW(V_prime):
        """Given V', set up linear system for U' and W'."""
        # Unknowns: U' has 8*12 = 96 entries, W' has 4*12 = 48 entries
        # But the constraint is bilinear in U and W, so this is nonlinear.
        # Instead, fix U' and solve for W' (linear in W')
        pass
    
    def verify_sub(U_prime, V_prime, W_prime):
        """Verify that the decomposition is correct."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s = 0
                    for r in range(target_rank):
                        s += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime[k*n_sub+i, r]
                    if s != 2:  # integer version: target is 2 (since 2*2*2=8... wait)
                        return False
        return True
    
    # Actually let me reconsider. With half-integer entries:
    # U, V, W have entries in {0, 1/2, 1, 3/2, ...}
    # Let U = U'/2, V = V'/2, W = W'/2 where U',V',W' are integers
    # Then sum_r (U'/2)(V'/2)(W'/2) = sum_r U'V'W'/8 = 1
    # So sum_r U'V'W' = 8
    
    def verify_sub_int(U_prime, V_prime, W_prime):
        """Verify integer version: sum U'V'W' = 8 for each (i,j,k)."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s = 0
                    for r in range(target_rank):
                        s += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime[k*n_sub+i, r]
                    if s != 8:
                        return False
        return True
    
    # Strategy: Try structured V patterns and solve linear system for W
    # Given U and V, the constraint for each (i,k) is:
    # sum_r W[k,i,r] * (sum_j U[i,j,r] * V[j,k,r]) = 8
    # This is linear in W! So for fixed U,V, we can solve for W.
    
    def try_UV(U_prime, V_prime):
        """Given U', V', solve linear system for W' and check if integer solution exists."""
        # For each (i,k): sum_r W[k,i,r] * C[i,k,r] = 8
        # where C[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        
        # W has 4*12 = 48 unknowns (4 positions: (k,i) for k in {0,1}, i in {0,1}, each with 12 values)
        # We have 4 equations (one per (i,k))
        # Underdetermined! We need to find ANY integer solution.
        
        # Actually wait - the constraint is for each (i,j,k), not just (i,k).
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 8 for EACH (i,j,k)
        
        # So for fixed (i,k), the constraint couples across j:
        # For fixed i,k: sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 8? No!
        # It's sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 8 for EACH j separately!
        
        # So we have 16 independent equations (one per (i,j,k)):
        # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 8
        
        # But W[k,i,r] appears in equations for different j. Let's group by (i,k):
        # For fixed (i,k) and varying j: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 8
        # This means: for each (i,k), the vector W[k,i,:] must satisfy:
        # sum_j [U[i,j,:] * V[j,k,:]] * W[k,i,:] = 8 (element-wise product then dot)
        
        # Let D[i,k,j,r] = U[i,j,r] * V[j,k,r]
        # Then: sum_{j,r} D[i,k,j,r] * W[k,i,r] = 8
        # Grouping: sum_r (sum_j D[i,k,j,r]) * W[k,i,r] = 8
        
        # So for each (i,k): sum_r C[i,k,r] * W[k,i,r] = 8
        # where C[i,k,r] = sum_j U[i,j,r]*V[j,k,r]
        
        # This gives 4 linear equations in the 48 W variables!
        # Very underdetermined. We just need any integer solution.
        
        C = np.zeros((n_sub, p_sub, target_rank), dtype=int)
        for i in range(n_sub):
            for k in range(p_sub):
                for r in range(target_rank):
                    C[i,k,r] = sum(U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] for j in range(m_sub))
        
        # For each (i,k): C[i,k,:] . W[k,i,:] = 8
        # Try to find integer W
        
        W_prime = np.zeros((p_sub * n_sub, target_rank), dtype=int)
        valid = True
        for i in range(n_sub):
            for k in range(p_sub):
                c_vec = C[i,k,:]
                if np.all(c_vec == 0):
                    # Can't satisfy the equation
                    valid = False
                    break
                # Find integer w such that c_vec . w = 8
                # Simple approach: try to put all weight on one component
                found = False
                for r in range(target_rank):
                    if c_vec[r] != 0 and 8 % c_vec[r] == 0:
                        W_prime[k*n_sub+i, r] = 8 // c_vec[r]
                        found = True
                        break
                if not found:
                    # Try two components
                    for r1 in range(target_rank):
                        if c_vec[r1] == 0:
                            continue
                        for r2 in range(target_rank):
                            if r2 <= r1 or c_vec[r2] == 0:
                                continue
                            # Solve: c1*w1 + c2*w2 = 8
                            # Try small integer values
                            for w1 in range(-4, 5):
                                rem = 8 - c_vec[r1]*w1
                                if rem % c_vec[r2] == 0:
                                    w2 = rem // c_vec[r2]
                                    if abs(w2) <= 4:
                                        W_prime[k*n_sub+i, r1] = w1
                                        W_prime[k*n_sub+i, r2] = w2
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        continue
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        # Verify
        if not verify_sub_int(U_prime, V_prime, W_prime):
            return None
        
        return W_prime
    
    # Now search over structured U, V patterns
    # Key insight: use the 4 j-indices symmetrically
    
    # Pattern 1: U depends only on (i, j mod 2) and V depends only on (j mod 2, k)
    # This reduces the problem significantly
    
    for attempt in range(3000):
        U_prime = np.zeros((n_sub * m_sub, target_rank), dtype=int)
        V_prime = np.zeros((m_sub * p_sub, target_rank), dtype=int)
        
        if attempt < 1000:
            # Structured: pair j=0 with j=2, j=1 with j=3
            for r in range(target_rank):
                # U[i,0,r] = U[i,2,r], U[i,1,r] = U[i,3,r]
                for i in range(n_sub):
                    u0 = random.choice([-2,-1,0,1,2])
                    u1 = random.choice([-2,-1,0,1,2])
                    U_prime[i*m_sub+0, r] = u0
                    U_prime[i*m_sub+2, r] = u0
                    U_prime[i*m_sub+1, r] = u1
                    U_prime[i*m_sub+3, r] = u1
                # V[0,k,r] = V[2,k,r], V[1,k,r] = V[3,k,r]
                for k in range(p_sub):
                    v0 = random.choice([-2,-1,0,1,2])
                    v1 = random.choice([-2,-1,0,1,2])
                    V_prime[0*p_sub+k, r] = v0
                    V_prime[2*p_sub+k, r] = v0
                    V_prime[1*p_sub+k, r] = v1
                    V_prime[3*p_sub+k, r] = v1
        elif attempt < 2000:
            # More general: random but sparse
            for r in range(target_rank):
                for i in range(n_sub):
                    for j in range(m_sub):
                        if random.random() < 0.5:
                            U_prime[i*m_sub+j, r] = random.choice([-2,-1,1,2])
                for j in range(m_sub):
                    for k in range(p_sub):
                        if random.random() < 0.5:
                            V_prime[j*p_sub+k, r] = random.choice([-2,-1,1,2])
        else:
            # Very sparse: 1-2 nonzeros per row per column
            for r in range(target_rank):
                for i in range(n_sub):
                    j_idx = random.sample(range(m_sub), random.randint(1,2))
                    for j in j_idx:
                        U_prime[i*m_sub+j, r] = random.choice([-2,-1,1,2])
                for k in range(p_sub):
                    j_idx = random.sample(range(m_sub), random.randint(1,2))
                    for j in j_idx:
                        V_prime[j*p_sub+k, r] = random.choice([-2,-1,1,2])
        
        W_prime = try_UV(U_prime, V_prime)
        if W_prime is not None:
            U_sol = (U_prime / 2.0).tolist()
            V_sol = (V_prime / 2.0).tolist()
            W_sol = (W_prime / 2.0).tolist()
            return U_sol, V_sol, W_sol
    
    return None

def _try_find_rank14_sub():
    """Try rank 14 for M_{2,4,2} as a fallback."""
    n_sub, m_sub, p_sub = 2, 4, 2
    target_rank = 14
    
    for _ in range(1000):
        U_prime = np.zeros((n_sub * m_sub, target_rank), dtype=int)
        V_prime = np.zeros((m_sub * p_sub, target_rank), dtype=int)
        
        for r in range(target_rank):
            nz_u = random.sample(range(n_sub * m_sub), random.randint(1,3))
            for idx in nz_u:
                U_prime[idx, r] = random.choice([-2, -1, 1, 2])
            nz_v = random.sample(range(m_sub * p_sub), random.randint(1,3))
            for idx in nz_v:
                V_prime[idx, r] = random.choice([-2, -1, 1, 2])
        
        A = np.zeros((n_sub * p_sub, n_sub * p_sub * target_rank))
        b = np.full(n_sub * p_sub, 8.0)
        
        for i in range(n_sub):
            for k in range(p_sub):
                eq_idx = i * p_sub + k
                for r in range(target_rank):
                    coeff = 0
                    for j in range(m_sub):
                        coeff += U_prime[i * m_sub + j, r] * V_prime[j * p_sub + k, r]
                    if coeff != 0:
                        w_idx = (k * n_sub + i) * target_rank + r
                        A[eq_idx, w_idx] = coeff
        
        W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        W_prime_sol = W_flat.reshape((n_sub * p_sub, target_rank))
        
        if not np.all(np.abs(W_prime_sol - np.round(W_prime_sol)) < 1e-6):
            continue
        
        W_prime_int = np.round(W_prime_sol).astype(int)
        
        valid = True
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    s_val = 0
                    for r in range(target_rank):
                        s_val += U_prime[i*m_sub+j, r] * V_prime[j*p_sub+k, r] * W_prime_int[k*n_sub+i, r]
                    if s_val != 8:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            U_sol = (U_prime / 2.0).tolist()
            V_sol = (V_prime / 2.0).tolist()
            W_sol = (W_prime_int / 2.0).tolist()
            return U_sol, V_sol, W_sol
            
    return None

def construct():
    start = time.time()
    deadline = start + 145.0
    
    result = _build_rank28()
    best_rank = len(result["U"][0])
    
    # Try to find a rank-12 sub-solution for rank-24 full solution
    if time.time() < deadline - 30:
        sub_sol = _try_find_rank12_sub()
        if sub_sol:
            U_sub, V_sub, W_sub = sub_sol
            # Construct full 2x4x4 solution with rank 24
            U_full = [[0] * 24 for _ in range(8)]
            V_full = [[0] * 24 for _ in range(16)]
            W_full = [[0] * 24 for _ in range(8)]
            
            # k=0,1 block (columns 0-11)
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+k][r] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[k*2+i][r] = W_sub[k*2+i][r]
                        
            # k=2,3 block (columns 12-23)
            for r in range(12):
                for i in range(2):
                    for j in range(4):
                        U_full[i*4+j][r+12] = U_sub[i*4+j][r]
                for j in range(4):
                    for k in range(2):
                        V_full[j*4+2+k][r+12] = V_sub[j*2+k][r]
                for k in range(2):
                    for i in range(2):
                        W_full[(2+k)*2+i][r+12] = W_sub[k*2+i][r]
            
            if _verify_decomposition(U_full, V_full, W_full):
                result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 24
                return result  # Early return with improved rank
    
    # Fallback: if we didn't find rank-24, try to verify and return rank-28
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        # Rebuild to be safe
        result = _build_rank28()
        best_rank = len(result["U"][0])
    
    return result
# EVOLVE-BLOCK-END
