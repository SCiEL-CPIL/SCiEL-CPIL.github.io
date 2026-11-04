"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def _verify(U, V, W, n, m, p, tol=1e-12):
        """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _schoolbook(n, m, p):
        """Return the schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    def _strassen_2x2_block(U, V, W, i0, j0, k0, col_start):
        """
        Add Strassen's 7-multiplication decomposition for a 2x2 * 2x2 -> 2x2 block.
        A is rows i0..i0+1, cols j0..j0+1
        B is rows j0..j0+1, cols k0..k0+1
        C is rows i0..i0+1, cols k0..k0+1
        Returns new column index.
        """
        col = col_start
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        def add_term(u_idx, v_idx, w_idx, coeff):
            nonlocal col
            U[u_idx][col] = coeff
            V[v_idx][col] = coeff
            W[w_idx][col] = coeff
            col += 1

        # M1: (A00+A11)*(B00+B11)
        # A00 = U[i0*m+j0], A11 = U[(i0+1)*m+(j0+1)]
        # B00 = V[j0*p+k0], B11 = V[(j0+1)*p+(k0+1)]
        # C00 = W[k0*n+i0], C11 = W[(k0+1)*n+(i0+1)]
        # We need: A00*B00, A00*B11, A11*B00, A11*B11 all with coefficient 1 in M1
        # But M1 contributes to C00 with +1 and C11 with +1
        # Actually Strassen: M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This is one rank-1 term where:
        # U contribution: A00 gets 1, A11 gets 1
        # V contribution: B00 gets 1, B11 gets 1
        # W contribution: C00 gets 1, C11 gets 1
        # But wait - this isn't a single rank-1 tensor. It's the product of sums.
        # In tensor decomposition, (A00+A11)(B00+B11)C00 means:
        # U[i0*m+j0] += 1, U[(i0+1)*m+(j0+1)] += 1
        # V[j0*p+k0] += 1, V[(j0+1)*p+(k0+1)] += 1
        # W[k0*n+i0] += 1
        # And similarly for C11.
        # But this is TWO rank-1 terms (one for C00, one for C11), not one!
        
        # Let me reconsider. In Strassen's algorithm:
        # M1 = (A00+A11)(B00+B11) - this is ONE scalar multiplication
        # It contributes to C00 and C11.
        # As a tensor decomposition term, M1 is:
        # (e_A00 + e_A11) ⊗ (e_B00 + e_B11) ⊗ (e_C00 + e_C11)
        # This is NOT a rank-1 tensor! It's rank 4.
        
        # The correct way: each M_i is ONE multiplication, giving ONE rank-1 term
        # where the U, V, W vectors have the appropriate coefficients.
        
        # M1: U has 1 at A00 and A11, V has 1 at B00 and B11, W has 1 at C00 and C11
        # But that's not rank-1. The rank-1 decomposition of (x+y)(z+w)(u+v) requires
        # 4 rank-1 terms in general.
        
        # I think I'm confusing myself. Let me be very explicit.
        # In Strassen, M1 = (A00+A11)*(B00+B11) is a single scalar product.
        # As a term in the tensor sum, it contributes:
        #   M1 * C00 and M1 * C11
        # So the tensor term is:
        #   (A00+A11) ⊗ (B00+B11) ⊗ C00  +  (A00+A11) ⊗ (B00+B11) ⊗ C11
        # = A00⊗B00⊗C00 + A00⊗B11⊗C00 + A11⊗B00⊗C00 + A11⊗B11⊗C00
        # + A00⊗B00⊗C11 + A00⊗B11⊗C11 + A11⊗B00⊗C11 + A11⊗B11⊗C11
        # That's 8 rank-1 terms, which is worse than schoolbook!
        
        # I see my error. Strassen's algorithm works at the LEVEL OF ALGORITHMS,
        # not at the level of tensor decomposition. The key insight is that M1 is
        # computed as a SINGLE multiplication of two (already summed) quantities.
        # In the tensor decomposition, this means we have a rank-1 term where:
        # U vector = (1 at A00, 1 at A11, 0 elsewhere)
        # V vector = (1 at B00, 1 at B11, 0 elsewhere)  
        # W vector = (1 at C00, 1 at C11, 0 elsewhere)
        # BUT this is NOT a valid rank-1 tensor term because the W vector has two
        # non-zero entries.
        
        # The correct interpretation: Strassen's algorithm has 7 multiplications,
        # but each multiplication's result is ADDED to potentially multiple C entries.
        # The tensor decomposition rank counts the number of scalar multiplications.
        # So M1 contributes ONE to the rank, and its effect on C00 and C11 is handled
        # by having the W factor have entries at both C00 and C11.
        
        # Wait, I think the issue is that in tensor decomposition:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # Each term u_c ⊗ v_c ⊗ w_c is rank 1, meaning u_c, v_c, w_c are vectors.
        # For M1: u_c has 1 at positions (i0,j0) and (i0+1,j0+1)
        #          v_c has 1 at positions (j0,k0) and (j0+1,k0+1)
        #          w_c has 1 at positions (k0,i0) and (k0+1,i0+1)
        # This IS a valid rank-1 term! The vectors can have multiple non-zero entries.
        # The product u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] gives:
        # For (i,j,k) = (i0,j0,k0): 1*1*1 = 1 (contributes to C00)
        # For (i,j,k) = (i0,j0,k0+1): 1*1*1 = 1 (contributes to C01? No...)
        
        # Let me re-examine the indexing. C[i,k] = sum_j A[i,j]*B[j,k]
        # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
        # W is indexed by k*n+i (the output position)
        
        # So for M1 to contribute to C00 = C[i0,k0] and C11 = C[i0+1,k0+1]:
        # w_c should have 1 at k0*n+i0 (for C00) and 1 at (k0+1)*n+(i0+1) (for C11)
        
        # Let's verify: T[i0,j0,k0] should get contribution from M1.
        # u_c[i0*m+j0] = 1, v_c[j0*p+k0] = 1, w_c[k0*n+i0] = 1 → product = 1 ✓
        # T[i0,j0,k0+1]: u_c[i0*m+j0]=1, v_c[j0*p+(k0+1)]=0 → product = 0 ✓
        # T[i0+1,j0,k0]: u_c[(i0+1)*m+j0]=0 → product = 0 ✓
        # T[i0+1,j0+1,k0+1]: u_c[(i0+1)*m+(j0+1)]=1, v_c[(j0+1)*p+(k0+1)]=1, w_c[(k0+1)*n+(i0+1)]=1 → product = 1 ✓
        # T[i0,j0+1,k0]: u_c[i0*m+(j0+1)]=0 → 0 ✓
        # T[i0+1,j0,k0+1]: u_c[(i0+1)*m+j0]=0 → 0 ✓
        # T[i0,j0+1,k0+1]: u_c[i0*m+(j0+1)]=0 → 0 ✓
        # T[i0+1,j0+1,k0]: v_c[(j0+1)*p+k0]=0 → 0 ✓
        
        # So M1 as a single rank-1 term works correctly!
        
        # Now let's implement all 7 Strassen terms:
        
        # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
        U[i0*m+j0][col] += 1.0
        U[(i0+1)*m+(j0+1)][col] += 1.0
        V[j0*p+k0][col] += 1.0
        V[(j0+1)*p+(k0+1)][col] += 1.0
        W[k0*n+i0][col] += 1.0
        W[(k0+1)*n+(i0+1)][col] += 1.0
        col += 1
        
        # M2 = (A10+A11)(B00), contributes to C10(+1), C11(-1)
        U[(i0+1)*m+j0][col] += 1.0
        U[(i0+1)*m+(j0+1)][col] += 1.0
        V[j0*p+k0][col] += 1.0
        W[k0*n+(i0+1)][col] += 1.0
        W[(k0+1)*n+(i0+1)][col] += -1.0
        col += 1
        
        # M3 = (A00)(B01-B11), contributes to C00(-1), C01(+1)
        U[i0*m+j0][col] += 1.0
        V[j0*p+(k0+1)][col] += 1.0
        V[(j0+1)*p+(k0+1)][col] += -1.0
        W[k0*n+i0][col] += -1.0
        W[(k0+1)*n+i0][col] += 1.0
        col += 1
        
        # M4 = (A11)(B10-B00), contributes to C00(+1), C10(+1)
        U[(i0+1)*m+(j0+1)][col] += 1.0
        V[(j0+1)*p+k0][col] += 1.0
        V[j0*p+k0][col] += -1.0
        W[k0*n+i0][col] += 1.0
        W[k0*n+(i0+1)][col] += 1.0
        col += 1
        
        # M5 = (A00+A01)(B11), contributes to C00(-1), C01(+1)
        U[i0*m+j0][col] += 1.0
        U[i0*m+(j0+1)][col] += 1.0
        V[(j0+1)*p+(k0+1)][col] += 1.0
        W[k0*n+i0][col] += -1.0
        W[(k0+1)*n+i0][col] += 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01), contributes to C11(+1)
        U[(i0+1)*m+j0][col] += 1.0
        U[i0*m+j0][col] += -1.0
        V[j0*p+k0][col] += 1.0
        V[j0*p+(k0+1)][col] += 1.0
        W[(k0+1)*n+(i0+1)][col] += 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11), contributes to C00(+1)
        U[i0*m+(j0+1)][col] += 1.0
        U[(i0+1)*m+(j0+1)][col] += -1.0
        V[(j0+1)*p+k0][col] += 1.0
        V[(j0+1)*p+(k0+1)][col] += 1.0
        W[k0*n+i0][col] += 1.0
        col += 1
        
        return col

    def _build_strassen_based(n, m, p):
        """
        Build a Strassen-based decomposition for 2x5*5x5.
        Split B into 2+2+1 blocks along both row and column dimensions.
        Split A's columns into 2+2+1 to match.
        Use Strassen for 2x2*2x2 blocks, schoolbook for 1x*1 blocks and cross terms.
        """
        rank = n * m * p  # start with max possible
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        
        # Block structure:
        # A: 2 rows x 5 cols → rows [0,1], cols [0,1,2,3,4]
        # B: 5 rows x 5 cols → rows [0,1,2,3,4], cols [0,1,2,3,4]
        # C: 2 rows x 5 cols → rows [0,1], cols [0,1,2,3,4]
        
        # Split columns of A / rows of B into: [0,1], [2,3], [4]
        # Split columns of B into: [0,1], [2,3], [4]
        
        # For each pair of (A-col-block, B-col-block):
        # A-col-block b_a gives A[:, b_a] (2 x size_a)
        # B-row-block b_a * B-col-block b_b gives B[b_a, b_b] (size_a x size_b)
        
        # Blocks:
        # A cols: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
        # B rows: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
        # B cols: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
        
        block_starts_a = [0, 2, 4]  # A column block starts
        block_sizes_a = [2, 2, 1]
        block_starts_b_row = [0, 2, 4]  # B row block starts
        block_sizes_b_row = [2, 2, 1]
        block_starts_b_col = [0, 2, 4]  # B col block starts
        block_sizes_b_col = [2, 2, 1]
        
        for ba in range(3):  # A col block / B row block
            for bb in range(3):  # B col block
                j0 = block_starts_a[ba]
                sa = block_sizes_a[ba]
                k0 = block_starts_b_col[bb]
                sb = block_sizes_b_col[bb]
                
                if sa == 2 and sb == 2:
                    # Use Strassen: 7 multiplications
                    col = _strassen_2x2_block(U, V, W, 0, j0, k0, col)
                elif sa == 2 and sb == 1:
                    # 2x1 * 1x1: 2 multiplications (schoolbook)
                    # A[0,j0]*B[j0,k0] -> C[0,k0]
                    # A[0,j0+1]*B[j0+1,k0] -> C[0,k0]  (wait, B row is just j0)
                    # Actually B block is 1x1: B[j0, k0]
                    # A block is 2x1: A[0,j0], A[1,j0]
                    # C[0,k0] += A[0,j0]*B[j0,k0]
                    # C[1,k0] += A[1,j0]*B[j0,k0]
                    # 2 rank-1 terms
                    U[0*m+j0][col] = 1.0
                    V[j0*p+k0][col] = 1.0
                    W[k0*n+0][col] = 1.0
                    col += 1
                    U[1*m+j0][col] = 1.0
                    V[j0*p+k0][col] = 1.0
                    W[k0*n+1][col] = 1.0
                    col += 1
                elif sa == 1 and sb == 2:
                    # 1x2 * 2x1: 2 multiplications
                    # A block is 1x2: A[0,j0], A[0,j0+1]  (but sa=1, so only A[0,j0])
                    # Wait, sa is the A col block size. If sa=1, A has 1 column.
                    # B block is 1x2: B[j0,k0], B[j0,k0+1]
                    # C[0,k0] += A[0,j0]*B[j0,k0]
                    # C[0,k0+1] += A[0,j0]*B[j0,k0+1]
                    # 2 rank-1 terms
                    U[0*m+j0][col] = 1.0
                    V[j0*p+k0][col] = 1.0
                    W[k0*n+0][col] = 1.0
                    col += 1
                    U[0*m+j0][col] = 1.0
                    V[j0*p+(k0+1)][col] = 1.0
                    W[(k0+1)*n+0][col] = 1.0
                    col += 1
                else:
                    # sa=1, sb=1: 1x1 * 1x1: 1 multiplication
                    U[0*m+j0][col] = 1.0
                    V[j0*p+k0][col] = 1.0
                    W[k0*n+0][col] = 1.0
                    col += 1
                    U[1*m+j0][col] = 1.0
                    V[j0*p+k0][col] = 1.0
                    W[k0*n+1][col] = 1.0
                    col += 1
        
        # Trim unused columns
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        return U, V, W, col

    def _try_hooi(r, n, m, p, max_iters=100, tol=1e-14):
        """Try to find rank-r decomposition via HOOI."""
        if time_left() < 10:
            return None
        
        best_err = float('inf')
        best_factors = None
        
        for trial in range(5):
            if time_left() < 10:
                break
            
            np.random.seed(1000 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            U /= np.linalg.norm(U) + 1e-15
            V /= np.linalg.norm(V) + 1e-15
            W /= np.linalg.norm(W) + 1e-15
            
            for it in range(max_iters):
                if time_left() < 10:
                    break
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_new[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
                
                V_new = np.zeros_like(V)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            V_new[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                
                W_new = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            W_new[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
                
                U, V, W = U_new, V_new, W_new
                
                if np.linalg.norm(U_new - U) < tol:
                    break
            
            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < best_err:
                best_err = max_err
                best_factors = (U.copy(), V.copy(), W.copy())
            
            if max_err < 1e-12:
                U_list = [[float(x) for x in U[row, :]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row, :]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row, :]] for row in range(p*n)]
                return U_list, V_list, W_list
        
        return None

    # First, try the Strassen-based construction
    if time_left() > 20:
        U_s, V_s, W_s, rank_s = _build_strassen_based(n, m, p)
        if _verify(U_s, V_s, W_s, n, m, p):
            return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try HOOI for lower ranks
    # Based on Strassen giving ~46, try from 40 upward
    for r in range(40, 47):
        if time_left() < 15:
            break
        result = _try_hooi(r, n, m, p)
        if result is not None:
            U_h, V_h, W_h = result
            if _verify(U_h, V_h, W_h, n, m, p):
                return {"U": U_h, "V": V_h, "W": W_h}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
