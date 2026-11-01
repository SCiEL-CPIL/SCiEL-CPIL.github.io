import time
import random

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    
    # Known valid rank-11 decomposition
    def get_rank11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # For each output element C[i][k] = sum_j A[i][j] * B[j][k]
        # We need: sum_r U[i*m+j][r] * V[j*p+k][r] == delta(i,k) for the tensor
        # Actually, the tensor is T[i,j,k] = delta(i,k) for matrix mult A*B
        # where A is n x m, B is m x p, C is n x p
        # C[i][k] = sum_j A[i][j] * B[j][k]
        # So T[i,j,k] should be 1 if we're computing C[i][k] from A[i][j]*B[j][k]
        # The decomposition: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should give
        # the correct coefficient for each (i,j,k)
        
        # Actually, for matrix multiplication tensor:
        # T[i,j,k] = 1 if i==k (for the standard convention where C[i,k] = sum_j A[i,j] B[j,k])
        # Wait, let me reconsider. The tensor for matrix mult is:
        # T_{i,j,k} where i indexes rows of A (and C), j indexes cols of A/rows of B, k indexes cols of B (and C)
        # T_{i,j,k} = 1 if we need A[i,j]*B[j,k] to contribute to C[i,k]
        # So T_{i,j,k} = 1 always? No, that's not right.
        
        # The matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
        # because every product A[i,j]*B[j,k] contributes to C[i,k]
        # Wait no, only when the indices match properly.
        
        # Let me think again. For C = A * B where A is n×m, B is m×p, C is n×p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
        # because each term A[i,j]*B[j,k] appears in exactly one output C[i,k]
        
        # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # (using the convention W[k*n+i, r])
        
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    best = get_rank11()
    assert verify(best["U"], best["V"], best["W"]), "Rank-11 verification failed"
    
    # Try to find rank-10 by merging the four pure terms (indices 7-10)
    # These correspond to:
    # Term 7: A[0,0]*B[0,2] -> C[2,0]  (i=0,j=0,k=2)
    # Term 8: A[0,1]*B[1,2] -> C[2,0]  (i=0,j=1,k=2)  
    # Term 9: A[1,0]*B[0,2] -> C[2,1]  (i=1,j=0,k=2)
    # Term 10: A[1,1]*B[1,2] -> C[2,1] (i=1,j=1,k=2)
    
    # We can try to represent these 4 terms with 3 or 2 terms using linear combinations
    # Strategy: use 2 terms with combined A and B factors
    
    # Try: replace terms 7-10 with 2 terms
    # Term A: (A[0,0] + A[1,1]) * (B[0,2] + B[1,2]) -> contributes to C[2,0] and C[2,1]
    # Term B: (A[0,1] + A[1,0]) * (B[0,2] - B[1,2]) -> contributes to C[2,0] and C[2,1]
    
    # Let's try various combinations
    
    def try_rank10():
        """Try to find a rank-10 decomposition."""
        base = get_rank11()
        U0, V0, W0 = base["U"], base["V"], base["W"]
        
        # Keep terms 0-6 (7 terms), try to replace terms 7-10 (4 terms) with 3 terms
        # Total: 7 + 3 = 10
        
        # The 4 terms we want to replace:
        # (i=0,j=0,k=2): U[0,0]=1, V[0,2]=1, W[2,0]=1
        # (i=0,j=1,k=2): U[0,1]=1, V[1,2]=1, W[2,0]=1
        # (i=1,j=0,k=2): U[1,0]=1, V[0,2]=1, W[2,1]=1
        # (i=1,j=1,k=2): U[1,1]=1, V[1,2]=1, W[2,1]=1
        
        # Try 3-term replacement
        # Term 7': (a00*A[0,0] + a01*A[0,1] + a10*A[1,0] + a11*A[1,1]) * (b00*B[0,2] + b01*B[1,2]) * (c00*W[2,0] + c01*W[2,1])
        # This is getting complex. Let's try a simpler approach.
        
        # Alternative: try to merge term 7 and 8 (both contribute to C[2,0])
        # and merge term 9 and 10 (both contribute to C[2,1])
        
        # For C[2,0]: A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # Can we write this as (x*A[0,0] + y*A[0,1]) * (z*B[0,2] + w*B[1,2])?
        # Expanding: xz*A[0,0]*B[0,2] + xw*A[0,0]*B[1,2] + yz*A[0,1]*B[0,2] + yw*A[0,1]*B[1,2]
        # We need: xz=1, yw=1, xw=0, yz=0
        # From xw=0 and yz=0: either x=0 or w=0, and either y=0 or z=0
        # If x=0, then xz=0≠1. If w=0, then xw=0 but we need xz=1 so x≠0, z≠0
        # If y=0, then yw=0≠1. If z=0, then yz=0 but we need yw=1 so y≠0, w≠0
        # So we can't do it with one term. Need at least 2 terms for each output.
        
        # But we can share terms between C[2,0] and C[2,1]!
        
        # Try:
        # Term 7': (A[0,0] + A[1,1]) * B[0,2] * (W[2,0] + W[2,1])
        #   Contributes: A[0,0]*B[0,2]*(W[2,0]+W[2,1]) + A[1,1]*B[0,2]*(W[2,0]+W[2,1])
        #   To C[2,0]: A[0,0]*B[0,2] + A[1,1]*B[0,2] (wrong, we only want A[0,0]*B[0,2])
        #   To C[2,1]: A[0,0]*B[0,2] + A[1,1]*B[0,2] (wrong)
        
        # This approach is tricky. Let's try a different strategy.
        
        # Use the fact that we can have negative coefficients.
        
        # Try 3 terms for the 4 pure products:
        # Term 7': (A[0,0] + A[1,1]) * (B[0,2] + B[1,2]) * (W[2,0] - W[2,1])
        # Term 8': (A[0,0] - A[1,1]) * (B[0,2] - B[1,2]) * (W[2,0] + W[2,1])
        # Term 9': (A[0,0] + A[1,1]) * (B[0,2] - B[1,2]) * W[2,0]
        # ... this is getting complicated.
        
        # Let's try a computational approach: solve for the coefficients.
        
        # We have 7 fixed terms (0-6) and want to add 3 more terms (7', 8', 9')
        # such that the total gives the correct tensor.
        
        # For each (i,j,k) with k=2, the contribution from terms 0-6 is:
        # Let's compute what terms 0-6 contribute to each (i,j,2)
        
        contrib_from_0_6 = {}
        for i in range(n):
            for j in range(m):
                for k in [2]:
                    s = 0
                    for r in range(7):
                        s += U0[i*m+j][r] * V0[j*p+k][r] * W0[k*n+i][r]
                    contrib_from_0_6[(i,j,k)] = s
        
        # For k=2, we need total contribution to be 1 for each (i,j,2)
        # So the 3 new terms must contribute 1 - contrib_from_0_6[(i,j,2)]
        
        # Check what we need:
        needed = {}
        for i in range(n):
            for j in range(m):
                k = 2
                needed[(i,j,k)] = 1 - contrib_from_0_6.get((i,j,k), 0)
        
        # Print for debugging
        # print("Needed contributions:", needed)
        
        # The 3 new terms have the form:
        # Term r (r=7,8,9): sum_{i',j'} U_new[i',j',r] * sum_{j',k'} V_new[j',k',r] * W_new[k',i',r]
        # For k=2, only V_new[j',2,r] matters
        # For each (i,j,2): contribution = sum_r U_new[i,j,r] * V_new[j,2,r] * W_new[2,i,r]
        
        # Let's denote:
        # a_r = U_new[0,0,r], b_r = U_new[0,1,r], c_r = U_new[1,0,r], d_r = U_new[1,1,r]
        # e_r = V_new[0,2,r], f_r = V_new[1,2,r]
        # g_r = W_new[2,0,r], h_r = W_new[2,1,r]
        
        # We need:
        # (0,0,2): a_r*e_r*g_r + ... = sum_r a_r*e_r*g_r = needed[(0,0,2)]
        # (0,1,2): sum_r b_r*f_r*g_r = needed[(0,1,2)]
        # (1,0,2): sum_r c_r*e_r*h_r = needed[(1,0,2)]
        # (1,1,2): sum_r d_r*f_r*h_r = needed[(1,1,2)]
        
        # And for k=0,1, the new terms should contribute 0:
        # For k=0: sum_r U_new[i,j,r] * V_new[j,0,r] * W_new[0,i,r] = 0 for all i,j
        # For k=1: sum_r U_new[i,j,r] * V_new[j,1,r] * W_new[1,i,r] = 0 for all i,j
        
        # This is a system of nonlinear equations. Let's try to solve it with integer/half-integer values.
        
        # Try small integer values for the coefficients.
        
        from itertools import product
        
        # Try values in {-2, -1, 0, 1, 2} for the 24 variables (3 terms * 8 vars each)
        # That's too many. Let's be smarter.
        
        # Assume symmetry: try to make the terms symmetric.
        
        # Try:
        # Term 7: U=(1,1,1,1), V=(1,1), W=(1,-1)  -> contributes (1,1,1,1) to (0,0,0,1)
        # Term 8: U=(1,-1,1,-1), V=(1,-1), W=(1,1)
        # Term 9: U=(1,1,-1,-1), V=(1,1), W=(1,1)
        
        # Let's just try a few structured patterns.
        
        patterns = [
            # (U00, U01, U10, U11, V02, V12, W20, W21) for each of 3 terms
            [
                (1, 0, 0, 1, 1, 1, 1, -1),
                (0, 1, 1, 0, 1, -1, 1, 1),
                (1, 0, 0, 1, 1, -1, -1, 1),
            ],
            [
                (1, 1, 1, 1, 1, 1, 1, -1),
                (1, -1, -1, 1, 1, -1, 1, 1),
                (1, 1, -1, -1, 1, 1, 1, 1),
            ],
            [
                (1, 0, 0, 1, 1, 1, 1, 0),
                (0, 1, 1, 0, 1, -1, 0, 1),
                (1, 0, 0, -1, 1, 1, 1, 1),
            ],
            [
                (1, 1, 0, 0, 1, 1, 1, 1),
                (0, 0, 1, 1, 1, -1, 1, -1),
                (1, 1, 1, 1, 1, 1, -1, 1),
            ],
        ]
        
        for pat in patterns:
            U_new = [[0]*3 for _ in range(n*m)]
            V_new = [[0]*3 for _ in range(m*p)]
            W_new = [[0]*3 for _ in range(p*n)]
            
            for t in range(3):
                u00, u01, u10, u11, v02, v12, w20, w21 = pat[t]
                U_new[0][t] = u00
                U_new[1][t] = u01
                U_new[2][t] = u10
                U_new[3][t] = u11
                V_new[2][t] = v02
                V_new[5][t] = v12
                W_new[4][t] = w20  # W[2*n+0] = W[4]
                W_new[5][t] = w21  # W[2*n+1] = W[5]
            
            # Build full matrices
            U = [row[:] for row in U0]
            V = [row[:] for row in V0]
            W = [row[:] for row in W0]
            
            # Extend to rank 10
            for i in range(n*m):
                U[i].extend(U_new[i])
            for i in range(m*p):
                V[i].extend(V_new[i])
            for i in range(p*n):
                W[i].extend(W_new[i])
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    if time.time() < deadline - 5:
        result = try_rank10()
        if result is not None:
            best = result
    
    return best
# EVOLVE-BLOCK-END
