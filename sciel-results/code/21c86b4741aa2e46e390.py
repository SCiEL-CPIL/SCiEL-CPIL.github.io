"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Try to find a rank-16 decomposition using Winograd on the 1x3x3 part
    # Strategy: Use 7 Strassen terms for A1(2x2)xB1_left(2x2), then try to
    # handle A1(2x2)xB1_right(2x1) and A2(2x1)xB2(1x3) with fewer terms
    # using Winograd F-algorithm transformations.
    
    # First, try a structured rank-16 search
    # The 1x3x1 tensor (vector outer product) has rank 3 (full rank).
    # For 2x1x3 (two vectors times 3-vector), the tensor is:
    #   T[a,b,c] = u_a * v_b * w_c for a in {0,1}, b in {0,1,2}, c in {0,1,2}
    # This is just 2 * rank-3 = rank 6 at most, but we can share.
    # Actually A2(2x1) x B2(1x3) gives C[0:2, 0:3] += A[i,2]*B[2,k]
    # This is outer product of column 2 of A with row 2 of B.
    # Rank of this sub-tensor: it's a 2x3 matrix outer product, rank = 2*3=6 if full.
    # But we can use Winograd to reduce.
    
    # Winograd F_2 for 1x2x1: (x0,x1)*(y0,y1) -> x0*y0, x1*y1
    # F_2 uses 3 terms: (x0+x1)*y0, (x0-x1)*y1, (x0+x1)*(y0-y1) - no wait
    # Standard F_2: m0=(x0+x1)*y0, m1=(x0-x1)*y1, m2=(x0+x1)*(y0-y1)
    # x0*y0 = (m0+m1+m2)/2, x1*y1 = (m0-m1+m2)/2... let me recheck
    # Actually: x0*y0 = (m0 + m1 + m2)/2? No.
    # m0 = (x0+x1)*y0 = x0*y0 + x1*y0
    # m1 = (x0-x1)*y1 = x0*y1 - x1*y1  
    # m2 = (x0+x1)*(y0-y1) = x0*y0 - x0*y1 + x1*y0 - x1*y1
    # m0 + m2 = 2*x0*y0 + 2*x1*y0 - x0*y1... this doesn't simplify nicely
    # 
    # Let me just use the standard Winograd for 2x2: 7 terms (Strassen)
    # For 1x3x1, the minimal rank is 3 (it's a full-rank 3x3 diagonal-like tensor)
    # So we can't save on individual 1x3x1.
    #
    # But for the combined problem, maybe we can share terms between
    # A1*B1_right and A2*B2.
    
    # Let me try a direct rank-16 search with structured approach.
    # Keep 7 Strassen terms, need 9 more for the remaining 9 direct terms.
    # If we can merge 2 pairs of direct terms into shared Winograd terms, we save 2.
    
    # Actually, let me try: use 7 Strassen + 9 direct = 16 total? 
    # 7 + 9 = 16! Let me check if the 9 direct terms can be exactly 9 rank-1 terms.
    # A1(2x2)x B1_right(2x1): 4 terms
    # A2(2x1) x B2(1x3): 6 terms
    # Total: 10 terms. 7+10=17. That's what we have.
    # To get 16, we need to save 1 term.
    
    # Idea: Can we combine one term from A1*B1_right with one from A2*B2?
    # A1*B1_right term: A[i,j]*B[j,2] for i in {0,1}, j in {0,1} -> C[i,2]
    # A2*B2 term: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2} -> C[i,k]
    # These involve different B indices (B[j,2] vs B[2,k]) and different A indices.
    # The only overlap is in the A matrix indices: A[0,0],A[0,1],A[1,0],A[1,1] vs A[0,2],A[1,2]
    # No overlap in A indices. So no direct sharing.
    
    # Alternative: Use a non-Strassen 2x2x2 decomposition with different structure
    # that allows sharing with the rest.
    
    # Let me try a random search for rank 16 with {-1,0,1} entries.
    # This is expensive but let's try a limited search.
    
    T = _build_tensor(n, m, p)
    
    # First, try a structured rank-16: 7 Strassen + 9 terms for the rest
    # where we try to merge 2 of the 10 direct terms.
    # 
    # Actually, let me try a different Strassen variant that uses the
    # "middle" row/column of B differently.
    
    # Let me try: what if we use a 3x3 Winograd on the full B matrix?
    # B is 3x3. Winograd F_3 for 1x3x1 needs 4 terms (vs 3 direct).
    # That's worse. But for 2x3x3, maybe there's a 15-term algorithm?
    
    # Known results: 2x3x3 matrix multiplication.
    # The trivial rank is 18. Strassen gives 7+4+6=17.
    # Is 16 known? Let me search computationally.
    
    # Try random search with small entries
    import random
    
    def _try_random_rank16(iterations=50000):
        """Try to find a rank-16 decomposition with entries in {-1,0,1}."""
        n_a, n_b, n_c = 6, 9, 6
        for it in range(iterations):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n_a, 16))
            V = np.zeros((n_b, 16))
            W = np.zeros((n_c, 16))
            
            for r in range(16):
                # Random sparse vectors with entries in {-1, 0, 1}
                for a in range(n_a):
                    U[a, r] = random.choice([-1, 0, 0, 1])
                for b in range(n_b):
                    V[b, r] = random.choice([-1, 0, 0, 1])
                for c in range(n_c):
                    W[c, r] = random.choice([-1, 0, 0, 1])
            
            if _verify_decomposition(U, V, W, T):
                return U, V, W
        
        return None
    
    # Try a more structured search: fix the Strassen part and search the rest
    def _try_structured_rank16():
        """Fix 7 Strassen terms, search for 9 terms for the remaining part."""
        n_a, n_b, n_c = 6, 9, 6
        
        # Set up Strassen terms (same as before)
        U = np.zeros((n_a, 16))
        V = np.zeros((n_b, 16))
        W = np.zeros((n_c, 16))
        
        def set_term(r, u_dict, v_dict, w_dict):
            for a, val in u_dict.items():
                U[a, r] = val
            for b, val in v_dict.items():
                V[b, r] = val
            for c, val in w_dict.items():
                W[c, r] = val
        
        # Strassen for A1(2x2) x B1_left(2x2)
        set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
        set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
        set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
        set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
        set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
        set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
        set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
        
        # Now we need to find 9 more terms (r=7..15) such that
        # sum_{r=7}^{15} U[:,r]*V[:,r]*W[:,r] = T - sum_{r=0}^{6} U[:,r]*V[:,r]*W[:,r]
        
        # Compute the residual
        residual = np.zeros((n_a, n_b, n_c))
        for r in range(7):
            residual += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        target = T - residual
        
        # The residual should only have non-zero entries where
        # A1*B1_right and A2*B2 contribute.
        # Let's verify the target structure
        # Non-zero entries in target:
        # From A1*B1_right: (a in {0,1,3,4}, b in {2,5}, c in {4,5})
        #   a=0,b=2,c=4; a=1,b=5,c=4; a=3,b=2,c=5; a=4,b=5,c=5
        # From A2*B2: (a in {2,5}, b in {6,7,8}, c in {0,1,2,3,4,5})
        #   a=2,b=6,c=0; a=2,b=7,c=2; a=2,b=8,c=4
        #   a=5,b=6,c=1; a=5,b=7,c=3; a=5,b=8,c=5
        
        # So target has exactly 10 non-zero entries, all = 1.
        # We need 9 rank-1 terms to produce these 10 unit entries.
        # This means at least one rank-1 term must produce 2 of the entries.
        
        # A rank-1 term U[:,r]*V[:,r]*W[:,r] produces entries at
        # (a,b,c) where U[a,r]!=0, V[b,r]!=0, W[c,r]!=0.
        # For it to produce exactly 2 of the target entries, it needs
        # to have support that covers exactly those 2 positions.
        
        # The 10 target positions are:
        # P1=(0,2,4), P2=(1,5,4), P3=(3,2,5), P4=(4,5,5)
        # P5=(2,6,0), P6=(2,7,2), P7=(2,8,4)
        # P8=(5,6,1), P9=(5,7,3), P10=(5,8,5)
        
        # For a rank-1 term to cover P1 and P2: U must have support at a=0,1
        # V must have support at b=2,5; W must have support at c=4.
        # But then it would also produce entries at (0,5,4) and (1,2,4) which
        # are NOT in the target. So this doesn't work directly.
        
        # Unless we use cancellations with other terms.
        
        # This is getting complex. Let me try a linear algebra approach.
        # We need to find 9 vectors (u_r, v_r, w_r) such that
        # sum_r u_r * v_r * w_r = target
        
        # This is a non-linear system. Let me try a different approach:
        # Fix the structure of some terms and solve for the rest.
        
        # Try: 4 terms for A1*B1_right (direct) and 5 terms for A2*B2 (Winograd-like)
        # A2*B2: A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0,1,2
        # This is: A[0,2]*(B[2,0]*C[0,0] + B[2,1]*C[0,1] + B[2,2]*C[0,2])
        #         + A[1,2]*(B[2,0]*C[1,0] + B[2,1]*C[1,1] + B[2,2]*C[1,2])
        
        # Can we do this in 5 terms? 
        # Let x0=A[0,2], x1=A[1,2], y0=B[2,0], y1=B[2,1], y2=B[2,2]
        # We need: x0*y0->C[0,0], x0*y1->C[0,1], x0*y2->C[0,2]
        #          x1*y0->C[1,0], x1*y1->C[1,1], x1*y2->C[1,2]
        # This is a 2x3 outer product. Rank = 6 (full). Can't do in 5.
        
        # What about combining A1*B1_right and A2*B2?
        # A1*B1_right: A[0,0]*B[0,2]->C[0,2], A[0,1]*B[1,2]->C[0,2]
        #              A[1,0]*B[0,2]->C[1,2], A[1,1]*B[1,2]->C[1,2]
        # A2*B2: A[0,2]*B[2,0]->C[0,0], A[0,2]*B[2,1]->C[0,1], A[0,2]*B[2,2]->C[0,2]
        #         A[1,2]*B[2,0]->C[1,0], A[1,2]*B[2,1]->C[1,1], A[1,2]*B[2,2]->C[1,2]
        
        # Notice C[0,2] gets contributions from A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2]
        # And C[1,2] gets contributions from A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2]
        
        # Can we use a Winograd-like trick on the B[*,2] column?
        # B[0,2], B[1,2], B[2,2] -> use F_3 (4 terms) instead of 3?
        # F_3 for 1x3: m0=(y0+y1+y2)*x0, m1=(y0-y1)*x1, m2=(y1-y2)*x2, m3=...
        # This doesn't obviously help.
        
        # Let me just try a random search with the constraint that
        # terms 7-15 only use the relevant indices.
        
        # Relevant A indices: {0,1,2,3,4,5} (all)
        # Relevant B indices: {2,5,6,7,8}
        # Relevant C indices: {0,1,2,3,4,5} (all)
        
        # But we want sparse terms. Let me try random search with
        # entries in {-1, 0, 1} but only on the relevant indices.
        
        a_indices = list(range(6))
        b_indices = [2, 5, 6, 7, 8]
        c_indices = list(range(6))
        
        best_result = None
        for it in range(20000):
            if time.time() > deadline:
                break
            
            U2 = np.zeros((n_a, 9))
            V2 = np.zeros((n_b, 9))
            W2 = np.zeros((n_c, 9))
            
            for r in range(9):
                # Random sparse vectors
                for a in a_indices:
                    U2[a, r] = random.choice([-1, 0, 0, 1])
                for b in b_indices:
                    V2[b, r] = random.choice([-1, 0, 0, 1])
                for c in c_indices:
                    W2[c, r] = random.choice([-1, 0, 0, 1])
            
            # Check if U2*V2*W2 = target
            result = np.zeros((n_a, n_b, n_c))
            for r in range(9):
                result += np.einsum('a,b,c->abc', U2[:,r], V2[:,r], W2[:,r])
            
            if np.allclose(result, target, atol=1e-9):
                U2_full = U2
                V2_full = V2
                W2_full = W2
                # Combine with Strassen
                U[7:16] = U2
                V[7:16] = V2
                W[7:16] = W2
                return U, V, W
        
        return None
    
    # Try structured search
    result = _try_structured_rank16()
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the known rank-17 solution
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Strassen for A1(2x2) x B1_left(2x2)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Direct for A1(2x2) x B1_right(2x1)
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Direct for A2(2x1) x B2(1x3)
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    # Verify
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
