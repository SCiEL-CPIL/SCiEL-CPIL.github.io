"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _strassen_2x2():
    """
    Return the 7 rank-1 factors for 2x2 matrix multiplication using Strassen's algorithm.
    Each factor is a tuple (u, v, w) where u is 4-dim, v is 4-dim, w is 4-dim.
    Indexing: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    u indexes (a,b,c,d), v indexes (e,f,g,h), w indexes (i,j,k,l)
    """
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(g+h)
    # M4 = (a+d)(h-f)
    # M5 = (a+c)(e-g)
    # M6 = (b-d)(f+h)
    # M7 = (b-c)(g)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    factors = []
    
    # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1] (contributes to C11 and C22)
    factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,-1]))
    
    # M2 = (c+d)(e): u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1] (contributes to C21 and C22)
    factors.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = (a-b)(g+h): u=[1,-1,0,0], v=[0,0,1,1], w=[0,1,0,1] (contributes to C12 and C22)
    factors.append(([1,-1,0,0], [0,0,1,1], [0,1,0,1]))
    
    # M4 = (a+d)(h-f): u=[1,0,0,1], v=[0,-1,0,1], w=[1,0,0,1] (contributes to C11 and C21)
    factors.append(([1,0,0,1], [0,-1,0,1], [1,0,0,1]))
    
    # M5 = (a+c)(e-g): u=[1,0,1,0], v=[1,0,-1,0], w=[-1,1,0,0] (contributes to C11 and C12)
    factors.append(([1,0,1,0], [1,0,-1,0], [-1,1,0,0]))
    
    # M6 = (b-d)(f+h): u=[0,1,0,-1], v=[0,1,0,1], w=[0,0,0,1] (contributes to C22)
    factors.append(([0,1,0,-1], [0,1,0,1], [0,0,0,1]))
    
    # M7 = (b-c)(g): u=[0,1,-1,0], v=[0,0,1,0], w=[-1,0,0,0] (contributes to C11)
    factors.append(([0,1,-1,0], [0,0,1,0], [-1,0,0,0]))
    
    return factors

def _build_kronecker_decomposition():
    """
    Build a rank-28 decomposition using Kronecker product structure.
    
    View 2x3 as 2x(2x2) where each column is a 2x2 block, and 3x8 as (2x2)x8.
    Actually, let's think of it differently:
    
    We have A (2x3) and B (3x8). Think of A as having 3 columns, each of size 2.
    Think of B as having 3 rows, each of size 8.
    
    Alternative: Partition the middle dimension (3) into blocks.
    
    Let's use a different approach. Think of:
    - A as 2 x 3, which we can view as 2 x (3)
    - B as 3 x 8
    
    The key insight: partition the 8 columns of B into 4 groups of 2.
    So B becomes 3 x (4 groups of 2).
    
    Then C = A @ B is 2 x 8, which is 2 x (4 groups of 2).
    
    For each group of 2 output columns, we compute a 2x2 matrix.
    The 2x3 matrix A multiplied by a 3x2 submatrix of B gives a 2x2 result.
    
    So we have 4 independent 2x2 multiplications, each involving:
    - A (2x3) and a 3x2 block of B
    - Result is 2x2
    
    For each 2x2 result, we can use Strassen if we first convert the 2x3 x 3x2 
    into a 2x2 x 2x2 multiplication.
    
    Actually, let's use a cleaner approach:
    
    Standard multiplication of 2x3 by 3x8 has rank 48.
    
    Let's partition differently. Write the 3 middle dimension as a sum.
    
    Better approach: Use the fact that we can do 2x3 * 3x8 by:
    1. Pad A to 2x4 (add a zero column)
    2. Pad B to 4x8 (add a zero row)
    3. Now we have 2x4 * 4x8
    
    Then view 2x4 as 2x(2x2) and 4x8 as (2x2)x8.
    
    Hmm, this is getting complicated. Let me use a direct construction.
    
    Actually, the cleanest approach:
    
    Rank-7 Strassen for 2x2: 7 multiplications
    Standard for 2x2: 4 multiplications
    
    If we can reduce 2x3 * 3x8 to some number of 2x2 * 2x2 multiplications,
    we can use Strassen for each.
    
    2x3 * 3x8: The middle dimension is 3. We can split 3 = 1 + 2.
    
    C = A[:,0] * B[0,:] + A[:,1:3] * B[1:3,:]
    
    First term: 2x1 * 1x8 = 2x8 (rank 16)
    Second term: 2x2 * 2x8 (rank 32)
    
    Total rank: 48. Not helpful.
    
    Let me try: 3 = 2 + 1
    C = A[:,:2] * B[:2,:] + A[:,2] * B[2,:]
    
    First: 2x2 * 2x8, Second: 2x1 * 1x8
    
    For 2x2 * 2x8: View as 4 multiplications of 2x2 * 2x2 (each output column pair)
    Using Strassen: 4 * 7 = 28
    
    For 2x1 * 1x8: rank 16
    
    Total: 28 + 16 = 44.
    
    But we can do better. Let's view 2x2 * 2x8 differently.
    
    2x2 * 2x8: The result is 2x8. We can compute this as:
    - Compute the 2x2 product once (rank 4 or 7 with Strassen)
    - Then multiply the 2x2 result by identity-like matrices to get 2x8?
    
    No, that's not right. 2x2 * 2x8 means:
    C = [[a,b],[c,d]] * [[e,f,g,h,i,j,k,l],[m,n,o,p,q,r,s,t]]
    
    This is 4 independent 2x2 * 2x2 multiplications (one for each pair of output columns).
    Each has rank 4 (standard) or 7 (Strassen).
    
    So 2x2 * 2x8 has rank 4*4 = 16 (standard) or 4*7 = 28 (Strassen).
    
    And 2x1 * 1x8 has rank 16.
    
    Total with Strassen: 28 + 16 = 44.
    Total with standard: 16 + 16 = 32.
    
    Wait, that's wrong. Let me recalculate.
    
    2x2 * 2x8:
    A = [[a,b],[c,d]], B = [[e,f,g,h,i,j,k,l],[m,n,o,p,q,r,s,t]]
    C = [[ae+bm, af+bn, ag+bo, ah+bp, ai+bq, aj+br, ak+bs, al+bt],
         [ce+dm, cf+dn, cg+do, ch+dp, ci+dq, cj+dr, ck+ds, cl+dt]]
    
    Each entry C[i,j] involves 2 multiplications (for 2x2*2x2) or 1 multiplication (for 2x1*1x8).
    
    For 2x2*2x2: rank 4 (standard) or 7 (Strassen)
    For 2x1*1x8: each entry is just a*b, so rank 1 per entry, 16 entries total = rank 16
    
    So 2x2*2x8 has rank 4*4 = 16 using standard, or 7*4 = 28 using Strassen.
    2x1*1x8 has rank 16.
    
    Total: 16+16 = 32 (all standard) or 28+16 = 44 (Strassen for 2x2 parts).
    
    But we can do better by combining!
    
    Actually, let's just build the rank-44 construction directly.
    
    Partition: C = A[:,:2] * B[:2,:] + A[:,2:3] * B[2:3,:]
    
    Part 1: 2x2 * 2x8
    Part 2: 2x1 * 1x8
    
    For Part 1, use Strassen for each 2x2*2x2 block (4 blocks, 7 mults each = 28)
    For Part 2, use standard (16 mults)
    
    Total: 44
    
    Let me build this.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll build rank-44 decomposition
    rank = 44
    
    U = np.zeros((n*m, rank))  # 6 x 44
    V = np.zeros((m*p, rank))  # 24 x 44
    W = np.zeros((p*n, rank))  # 16 x 44
    
    # Part 1: 2x2 * 2x8 (columns 0,1 of A, rows 0,1 of B)
    # A[:,:2] is 2x2, B[:2,:] is 2x8
    # Split B[:2,:] into 4 blocks of 2x2
    # For each block, use Strassen (7 mults)
    
    strassen_factors = _strassen_2x2()
    
    col_offset = 0
    for block_idx in range(4):
        # Block covers columns 2*block_idx to 2*block_idx+1 of B
        # And columns 2*block_idx to 2*block_idx+1 of C
        for s in range(7):
            u_str, v_str, w_str = strassen_factors[s]
            r = col_offset + s
            
            # u_str indexes [a,b,c,d] of the 2x2 A block
            # A block is A[:,:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
            # u_str[0] -> A[0,0], u_str[1] -> A[0,1], u_str[2] -> A[1,0], u_str[3] -> A[1,1]
            U[0*2+0, r] += u_str[0]  # A[0,0]
            U[0*2+1, r] += u_str[1]  # A[0,1]
            U[1*2+0, r] += u_str[2]  # A[1,0]
            U[1*2+1, r] += u_str[3]  # A[1,1]
            
            # v_str indexes [e,f,g,h] of the 2x2 B block
            # B block is B[:2, 2*block_idx:2*block_idx+2]
            # = [[B[0,2*block_idx], B[0,2*block_idx+1]],
            #    [B[1,2*block_idx], B[1,2*block_idx+1]]]
            # v_str[0] -> B[0,2*block_idx], v_str[1] -> B[0,2*block_idx+1]
            # v_str[2] -> B[1,2*block_idx], v_str[3] -> B[1,2*block_idx+1]
            V[0*8 + 2*block_idx + 0, r] += v_str[0]  # B[0, 2*block_idx]
            V[0*8 + 2*block_idx + 1, r] += v_str[1]  # B[0, 2*block_idx+1]
            V[1*8 + 2*block_idx + 0, r] += v_str[2]  # B[1, 2*block_idx]
            V[1*8 + 2*block_idx + 1, r] += v_str[3]  # B[1, 2*block_idx+1]
            
            # w_str indexes [i,j,k,l] of the 2x2 C block
            # C block is C[:, 2*block_idx:2*block_idx+2]
            # = [[C[0,2*block_idx], C[0,2*block_idx+1]],
            #    [C[1,2*block_idx], C[1,2*block_idx+1]]]
            # w_str[0] -> C[0,2*block_idx], w_str[1] -> C[0,2*block_idx+1]
            # w_str[2] -> C[1,2*block_idx], w_str[3] -> C[1,2*block_idx+1]
            # C index: k*n+i where k is column, i is row
            W[(2*block_idx+0)*2 + 0, r] += w_str[0]  # C[0, 2*block_idx]
            W[(2*block_idx+0)*2 + 1, r] += w_str[1]  # C[0, 2*block_idx+1]
            W[(2*block_idx+1)*2 + 0, r] += w_str[2]  # C[1, 2*block_idx]
            W[(2*block_idx+1)*2 + 1, r] += w_str[3]  # C[1, 2*block_idx+1]
    
    col_offset += 28
    
    # Part 2: 2x1 * 1x8 (column 2 of A, row 2 of B)
    # A[:,2] is 2x1, B[2,:] is 1x8
    # C contribution: A[:,2] * B[2,:] which is 2x8
    # Each entry C[i,k] += A[i,2] * B[2,k]
    for k in range(8):
        r = col_offset + k
        # U: A[i,2] for i=0,1
        U[0*2+2, r] = 1  # A[0,2]
        U[1*2+2, r] = 1  # A[1,2]
        
        # V: B[2,k]
        V[2*8 + k, r] = 1  # B[2,k]
        
        # W: C[i,k] for i=0,1
        W[k*2 + 0, r] = 1  # C[0,k]
        W[k*2 + 1, r] = 1  # C[1,k]
    
    col_offset += 16
    
    assert col_offset == rank, f"Expected {rank}, got {col_offset}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    # Try the Kronecker/Strassen construction (rank 44)
    U_kr, V_kr, W_kr = _build_kronecker_decomposition()
    if _verify(U_kr, V_kr, W_kr, tol=1e-9):
        best_U, best_V, best_W = U_kr, V_kr, W_kr
        best_rank = 44
    
    return {"U": best_U, "V": best_V, "W": best_W}
