"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Tensor: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # We want rank-r decomposition: T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c_out,c]
    # where W is indexed by [c_out, c] with c_out = k*n+i
    
    # Non-zero entries of T:
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Try rank 20 decomposition using structured ansatz
    # Strategy: Use the polynomial embedding / Cohn-Winograd idea
    # 
    # For 2x2 * 2x6:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write this as: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # 
    # Let's use the following structure:
    # For each k in 0..5, we need 2 outputs (i=0 and i=1)
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is like 6 independent 2x2 matrix multiplications (each 1x2 * 2x1)
    # A 2x2 * 2x1 can be done in rank 3 (Strassen-like) or rank 4 (schoolbook)
    # But they share the B factors...
    #
    # Actually each "slice" k gives a 2x1 output from 2x2 * 2x1.
    # The 2x2 * 2x1 tensor has rank 4 (schoolbook) but can be done in rank 3?
    # No, 2x2*2x1 is just 4 bilinear terms, rank 4.
    #
    # But across all k, the A factors are shared. So we have:
    # 12 outputs, each is a linear combination of 4 products.
    #
    # Let's try a rank-20 construction:
    # Use 4 terms for the "diagonal" structure and 16 for the rest.
    
    # Let me try a direct numerical solve for rank 20
    rank = 20
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Build the constraint matrix: for each (a,b,c) in nonzero, 
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1
    # For each (a,b,c) NOT in nonzero, sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a system of polynomial equations. Let's use a structured initialization.
    
    # Ansatz: Use the following pattern based on the algebraic structure
    # U has 4 rows, V has 12 rows, W has 12 rows, each with 20 columns
    
    # Let's use a randomized approach with verification
    import random
    random.seed(42)
    
    best_rank = None
    best_decomp = None
    
    for target_rank in [20, 22, 24, 26, 28, 30]:
        # Initialize with sparse random
        U = np.zeros((dim_u, target_rank))
        V = np.zeros((dim_v, target_rank))
        W = np.zeros((dim_w, target_rank))
        
        # Assign each nonzero entry to a specific rank column
        # 24 nonzero entries, target_rank columns
        # Each column can handle multiple (a,b,c) tuples if they're compatible
        
        # For column r, it contributes U[a,r]*V[b,r]*W[c,r] to T[a,b,c]
        # For it to contribute to multiple nonzero entries, those entries must share
        # the same a, same b, or same c pattern... actually they need to be "compatible"
        
        # Simple assignment: one nonzero per column, with 24 columns used, rest zero
        # But that gives rank 24. For rank 20, we need some columns to handle 2+ entries.
        
        # A column r can handle entries (a1,b1,c1) and (a2,b2,c2) if:
        # U[a1,r]*V[b1,r]*W[c1,r] + ... and we can set values to satisfy both
        # This requires that the "pattern" allows it.
        
        # Let's try: for each k, handle the 2 entries (i=0,k) and (i=1,k) together
        # Entry for (i,j,k): a=i*2+j, b=j*6+k, c=k*2+i
        # For fixed k, j=0: (a=i, b=k, c=k*2+i) for i=0,1
        # For fixed k, j=1: (a=i+2, b=6+k, c=k*2+i) for i=0,1
        #
        # So for fixed k, we have 4 entries:
        # (0, k, 2k), (1, k, 2k+1), (2, 6+k, 2k), (3, 6+k, 2k+1)
        #
        # Can we handle these 4 with 2 rank-1 terms?
        # Term 1: U[0,r]*V[k,r]*W[2k,r] = 1, U[1,r]*V[k,r]*W[2k+1,r] = 1,
        #          U[2,r]*V[6+k,r]*W[2k,r] = 1, U[3,r]*V[6+k,r]*W[2k+1,r] = 1
        # And zero for all other (a,b,c)
        #
        # From first: U[0,r]*V[k,r]*W[2k,r] = 1
        # From third: U[2,r]*V[6+k,r]*W[2k,r] = 1
        # Divide: U[2,r]/U[0,r] = V[k,r]/V[6+k,r]
        #
        # From second: U[1,r]*V[k,r]*W[2k+1,r] = 1
        # From fourth: U[3,r]*V[6+k,r]*W[2k+1,r] = 1
        # Divide: U[3,r]/U[1,r] = V[k,r]/V[6+k,r]
        #
        # So we need U[2,r]/U[0,r] = U[3,r]/U[1,r] = V[k,r]/V[6+k,r] = some ratio t_k
        #
        # This is feasible! Set:
        # U[0,r]=1, U[2,r]=t_k, U[1,r]=1, U[3,r]=t_k
        # V[k,r]=1, V[6+k,r]=1/t_k
        # W[2k,r]=1, W[2k+1,r]=1
        #
        # Check: U[0,r]*V[k,r]*W[2k,r] = 1*1*1 = 1 ✓
        #         U[2,r]*V[6+k,r]*W[2k,r] = t_k*(1/t_k)*1 = 1 ✓
        #         U[1,r]*V[k,r]*W[2k+1,r] = 1*1*1 = 1 ✓
        #         U[3,r]*V[6+k,r]*W[2k+1,r] = t_k*(1/t_k)*1 = 1 ✓
        #
        # Now check zeros: For any (a,b,c) not in the 4 entries above,
        # U[a,r]*V[b,r]*W[c,r] must be 0.
        #
        # U is nonzero only at rows 0,1,2,3 (all 4 rows are nonzero in this term)
        # V is nonzero only at rows k and 6+k
        # W is nonzero only at rows 2k and 2k+1
        #
        # So the product is nonzero only when:
        # a in {0,1,2,3}, b in {k, 6+k}, c in {2k, 2k+1}
        #
        # The 4 nonzero entries are exactly:
        # (0,k,2k), (0,k,2k+1), (1,k,2k), (1,k,2k+1), (2,6+k,2k), (2,6+k,2k+1), (3,6+k,2k), (3,6+k,2k+1)
        #
        # But we only want 4 of these to be 1, and the other 4 to be 0!
        #
        # Specifically:
        # (0,k,2k)=1, (1,k,2k+1)=1, (2,6+k,2k)=1, (3,6+k,2k+1)=1
        # But (0,k,2k+1), (1,k,2k), (2,6+k,2k+1), (3,6+k,2k) should be 0
        #
        # With our current setup:
        # (0,k,2k+1) = U[0,r]*V[k,r]*W[2k+1,r] = 1*1*1 = 1 ≠ 0 ✗
        
        # So this doesn't work directly. We need a different structure.
        
        # Alternative: Use 4 terms per k, one for each entry. That's 24 terms = rank 24.
        # To get rank 20, we need to share some terms.
        
        # Let's try: for each k, use 3 terms instead of 4.
        # Can we handle 4 entries with 3 rank-1 terms?
        
        # This is getting complex. Let me just do rank 24 (schoolbook) and verify,
        # then try to optimize.
        
        if target_rank == 24:
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                        col += 1
            # Verify
            valid = True
            for a in range(dim_u):
                for b in range(dim_v):
                    for c in range(dim_w):
                        expected = 1 if (a, b, c) in nonzero else 0
                        actual = 0
                        for r in range(target_rank):
                            actual += U[a, r] * V[b, r] * W[c, r]
                        if abs(actual - expected) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
    
    if best_decomp is None:
        # Fallback: schoolbook rank 24
        rank = 24
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_decomp = (U, V, W)
    
    U, V, W = best_decomp
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Check if integer or half-integer
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
