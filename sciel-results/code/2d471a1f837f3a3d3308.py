import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
DEADLINE = 150
SAFETY_MARGIN = 10

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Best result found so far (schoolbook)
    best_rank = rank_schoolbook
    best_result = None

    # Construct schoolbook as fallback
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        for row in U:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in V:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in W:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_round_and_verify(U_float, V_float, W_float, rank):
        """Try rounding to half-integers and verify."""
        U_r = [[round(x * 2) / 2 for x in row] for row in U_float]
        V_r = [[round(x * 2) / 2 for x in row] for row in V_float]
        W_r = [[round(x * 2) / 2 for x in row] for row in W_float]
        
        if verify(U_r, V_r, W_r, rank):
            return U_r, V_r, W_r
        return None

    # Construct rank-33 decomposition using Strassen on 2x2 core blocks
    # 
    # Strategy: Split the 3 columns of A into:
    #   - 2x2 core (columns 0,1) x each 2-column block of B (3 blocks) -> use Strassen (rank 7 each)
    #   - 1x3 tail (column 2) x all 6 columns of B -> schoolbook (rank 12)
    # Total rank: 3*7 + 12 = 33
    #
    # For Strassen on 2x2x2:
    # A2 = [[a0, a1], [b0, b1]], B2 = [[c0, c1], [d0, d1]]
    # C2 = A2 * B2 = [[c00, c01], [c10, c11]]
    #
    # Strassen's 7 products:
    # M1 = (a0+a1)*(d0-d1)
    # M2 = (a0+a1)*d1
    # M3 = a0*(c0-d0)
    # M4 = b1*(c0+c1)
    # M5 = (b0+b1)*(c0-d0)
    # M6 = (a1-b1)*(c0+c1)
    # M7 = (a0-b0)*(d0+d1)
    #
    # c00 = M2 + M4 - M5 - M7
    # c01 = M1 + M5
    # c10 = M3 + M7
    # c11 = M1 - M2 + M3 + M4
    #
    # In tensor form for each Strassen term, the (i,j,k) coefficient is:
    # U[i*m+j] * V[j*p+k] * W[k*n+i]
    # where i is row of A (0 or 1), j is col of A (0 or 1), k is col of B (0 or 1)

    def make_strassen_2x2x2(offset_i, offset_j, offset_k, base_col):
        """
        Generate the 7 Strassen tensor terms for a 2x2x2 subproblem.
        offset_i: row offset in A (which row of the 2x3 matrix)
        offset_j: col offset in A (which col of the 2x3 matrix)  
        offset_k: col offset in B (which col of the 3x6 matrix)
        base_col: starting column index in the factor matrices
        Returns list of (U_vals, V_vals, W_vals) for each of the 7 terms.
        """
        terms = []
        
        # Strassen M1 = (a0+a1)*(d0-d1)
        # Contribution to c01 and c11
        # For c01: coefficient of a0*d0 is 0, a0*d1 is -1, a1*d0 is 1, a1*d1 is -1
        # Wait, let me be more careful.
        # M1 = (a0+a1)*(d0-d1) = a0*d0 - a0*d1 + a1*d0 - a1*d1
        # c01 = M1 + M5
        # c11 = M1 - M2 + M3 + M4
        #
        # The tensor term for M1 contributes to:
        #   C[0,1]: from a0*d0 (coeff +1), a0*d1 (coeff -1), a1*d0 (coeff +1), a1*d1 (coeff -1)
        #   C[1,1]: from a0*d0 (coeff +1), a0*d1 (coeff -1), a1*d0 (coeff +1), a1*d1 (coeff -1)
        #
        # Actually, in the tensor decomposition, each rank-1 term is:
        # sum over (i,j,k) of u[i,j] * v[j,k] * w[k,i]
        #
        # For M1 = (a0+a1)*(d0-d1):
        # This is a product of two linear forms, so it's rank 1 in the tensor sense.
        # u[0,0]=1, u[0,1]=0, u[1,0]=1, u[1,1]=0  (a0+a1)
        # v[0,0]=1, v[0,1]=-1, v[1,0]=0, v[1,1]=0  (d0-d1)
        # w[0,0]=0, w[0,1]=1, w[1,0]=1, w[1,1]=0  (contributes to c01 and c11)
        #
        # But wait - this single rank-1 term gives:
        # C[0,1] += (a0+a1)*(d0-d1)*w[1,0] = (a0+a1)*(d0-d1)*1
        # C[1,1] += (a0+a1)*(d0-d1)*w[0,1] = (a0+a1)*(d0-d1)*1
        #
        # Hmm, that's not quite right. Let me think about this differently.
        #
        # The matrix multiplication tensor for 2x2x2 has entries T[i,j,k] = delta (we want sum = 1 for each (i,j,k) that appears in the product).
        # Actually no - the tensor T[i,j,k] represents the coefficient such that C[i,k] = sum_{j} sum_{terms} ... 
        #
        # Let me restart with the standard formulation:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: C[i,k] = sum_{t=1}^{r} U[i,j_t] * V[j_t,k] * W[k,i] ... no
        #
        # The standard tensor decomposition: T[i,j,k] = sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t]
        # where T[i,j,k] = 1 if (i,j,k) is a valid triple (always is for matrix mult), 0 otherwise.
        # Wait no. For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k), and we need:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_t U[i,j,t] V[j,k,t] W[k,i,t] ... 
        #
        # Actually the correct formulation: The bilinear map (A,B) -> C has C[i,k] = sum_j A[i,j] B[j,k].
        # A rank-r decomposition means: C[i,k] = sum_{t=1}^{r} (sum_j A[i,j] U[i,j,t]) * (sum_l B[l,k] V[l,k,t]) * W[k,i,t]
        # No wait, that's not standard either.
        #
        # Standard: C[i,k] = sum_{t=1}^{r} U_t[i,j] V_t[j,k] W_t[k,i] where the sum over t replaces the sum over j.
        # More precisely: C[i,k] = sum_t (sum_j A[i,j] * u_t[i,j]) * (sum_l B[l,k] * v_t[l,k]) * w_t[k,i]
        # Hmm, this is getting confused. Let me use the explicit form from the verify function:
        #
        # verify checks: for each (i,j,k): sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
        #
        # So the tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1.
        # This means we're decomposing the all-ones tensor of shape (n*m) x (m*p) x (p*n) into rank-r terms.
        #
        # For Strassen on 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
        # We need T[i,j,k] = 1 for all 8 entries.
        #
        # Strassen's algorithm computes C = A*B where A,B are 2x2.
        # The 7 intermediate products are bilinear in A and B.
        # Each product M_t = (linear form in A) * (linear form in B).
        # Then C[i,k] = sum_t coeff_{i,k,t} * M_t.
        #
        # In the tensor decomposition framework:
        # M_t = (sum_j A[i,j] * u_t[i,j]) * (sum_l B[l,k] * v_t[l,k])
        # C[i,k] = sum_t w_t[k,i] * M_t
        #
        # So: T[i,j,k] = sum_t u_t[i,j] * v_t[l,k] * w_t[k,i] ... but this has a free index l.
        #
        # I think the correct mapping is:
        # M_t = (sum_j A[i,j] u_t[i,j]) * (sum_k B[j,k] v_t[j,k])
        # C[i,k] = sum_t w_t[k,i] * M_t = sum_t w_t[k,i] * (sum_j A[i,j] u_t[i,j]) * (sum_k' B[j,k'] v_t[j,k'])
        #
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_t w_t[k,i] * u_t[i,j] * v_t[j,k] = 1 for all (i,j,k)
        #
        # So the tensor is T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] where:
        # U[i,j,t] = u_t[i,j], V[j,k,t] = v_t[j,k], W[k,i,t] = w_t[k,i]
        #
        # OK so now for Strassen:
        # A = [[a0, a1], [b0, b1]], B = [[c0, c1], [d0, d1]]
        # (using row-major: A[0,0]=a0, A[0,1]=a1, A[1,0]=b0, A[1,1]=b1)
        # B[0,0]=c0, B[0,1]=c1, B[1,0]=d0, B[1,1]=d1
        
        # M1 = (a0+a1)*(d0-d1)
        # u1: u1[0,0]=1, u1[0,1]=1, u1[1,0]=0, u1[1,1]=0
        # v1: v1[0,0]=0, v1[0,1]=0, v1[1,0]=1, v1[1,1]=-1
        # w1: contributes to C[0,1] and C[1,1]
        # C[0,1] = M1 + M5, so w1[1,0] = 1 (for C[0,1])
        # C[1,1] = M1 - M2 + M3 + M4, so w1[0,1] = 1 (for C[1,1])
        # w1[0,0] = 0, w1[1,0] = 1, w1[0,1] = 1, w1[1,1] = 0
        
        # Wait, I need to be more careful with indices.
        # C[i,k] where i is row of C (= row of A), k is col of C (= col of B)
        # W[k*n+i] in the verify function, where n=2 for the 2x2 subproblem
        # So W[k*2 + i]
        #
        # C[0,0] = M2 + M4 - M5 - M7
        # C[0,1] = M1 + M5
        # C[1,0] = M3 + M7
        # C[1,1] = M1 - M2 + M3 + M4
        #
        # For each M_t, w_t[k,i] is the coefficient of M_t in C[i,k]:
        # M1: w[0,0]=0, w[1,0]=1, w[0,1]=1, w[1,1]=0  -> W[0*2+0]=0, W[1*2+0]=1, W[0*2+1]=1, W[1*2+1]=0
        #     i.e., W[k*2+i]: W[0,0]=0, W[1,0]=1, W[0,1]=1, W[1,1]=0
        # M2: w[0,0]=1, w[1,0]=0, w[0,1]=-1, w[1,1]=0
        # M3: w[0,0]=0, w[1,0]=0, w[0,1]=1, w[1,1]=1
        # M4: w[0,0]=1, w[1,0]=0, w[0,1]=0, w[1,1]=1
        # M5: w[0,0]=-1, w[1,0]=1, w[0,1]=0, w[1,1]=0
        # M6: (unused in standard Strassen output? Let me recheck)
        # M7: w[0,0]=-1, w[1,0]=0, w[0,1]=0, w[1,1]=1
        #
        # Wait, M6 is not used in the standard 4 output entries? Let me recheck Strassen:
        # M1 = (A00+A11)(B01+B11) -- no, different conventions exist
        #
        # Let me use the standard Strassen formulas:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b01 - b11)
        # M4 = a11(b10 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        #
        # OK let me use this convention. A[i,j], B[j,k] with i,j,k in {0,1}.
        # A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11
        # B[0,0]=b00, B[0,1]=b01, B[1,0]=b10, B[1,1]=b11
        
        # M1 = (a00+a11)(b00+b11)
        # u1: [1,0, 0,1] (a00 + a11)
        # v1: [1,0, 0,1] (b00 + b11)
        # w1: c00 coeff=1, c01=0, c10=0, c11=1 -> W[0,0]=1, W[1,0]=0, W[0,1]=0, W[1,1]=1
        #     In W[k*2+i]: W[0]=1, W[1]=0, W[2]=0, W[3]=1
        
        # M2 = (a10+a11)b00
        # u2: [0,0, 1,1]
        # v2: [1,0, 0,0]
        # w2: c00=0, c01=0, c10=1, c11=-1 -> W[0]=0, W[1]=1, W[2]=0, W[3]=-1
        
        # M3 = a00(b01-b11)
        # u3: [1,0, 0,0]
        # v3: [0,1, 0,-1]
        # w3: c00=0, c01=1, c10=0, c11=1 -> W[0]=0, W[1]=0, W[2]=1, W[3]=1
        
        # M4 = a11(b10-b00)
        # u4: [0,0, 0,1]
        # v4: [0,0, 1,-1]  -- wait, b10 - b00 means B[1,0] - B[0,0]
        #     v4[0,0]=-1, v4[0,1]=0, v4[1,0]=1, v4[1,1]=0
        # w4: c00=1, c01=0, c10=1, c11=0 -> W[0]=1, W[1]=0, W[2]=0, W[3]=0
        
        # M5 = (a00+a01)b11
        # u5: [1,1, 0,0]
        # v5: [0,0, 0,1]
        # w5: c00=-1, c01=1, c10=0, c11=0 -> W[0]=-1, W[1]=0, W[2]=1, W[3]=0
        
        # M6 = (a10-a00)(b00+b01)
        # u6: [-1,0, 1,0]
        # v6: [1,1, 0,0]
        # w6: c00=0, c01=0, c10=0, c11=1 -> W[0]=0, W[1]=0, W[2]=0, W[3]=1
        
        # M7 = (a01-a11)(b10+b11)
        # u7: [0,1, 0,-1]
        # v7: [0,0, 1,1]
        # w7: c00=1, c01=0, c10=0, c11=0 -> W[0]=1, W[1]=0, W[2]=0, W[3]=0
        
        # Now map to the verify function's indexing:
        # U[i*m+j] where m=3 for the full problem, but for the 2x2 sub-block:
        # The 2x2 sub-block uses rows 0,1 of A and cols offset_j, offset_j+1
        # V[j*p+k] where p=6, uses cols offset_k, offset_k+1 of B
        # W[k*n+i] where n=2, uses rows offset_i, offset_i+1 of C... 
        # Wait, n=2 is the number of rows of A, which is the number of rows of C.
        # For the 2x2 sub-block, i is in {0,1} (local), mapped to global row i.
        # j is in {0,1} (local), mapped to global col offset_j + j.
        # k is in {0,1} (local), mapped to global col offset_k + k.
        
        # U global index: i * m + (offset_j + j) = i * 3 + offset_j + j
        # V global index: (offset_j + j) * p + (offset_k + k) = (offset_j + j) * 6 + offset_k + k
        # W global index: (offset_k + k) * n + i = (offset_k + k) * 2 + i
        
        # Strassen terms (local i,j,k in {0,1}):
        # u_vals: list of 4 values [u[0,0], u[0,1], u[1,0], u[1,1]]
        # v_vals: list of 4 values [v[0,0], v[0,1], v[1,0], v[1,1]]
        # w_vals: list of 4 values [w[0,0], w[1,0], w[0,1], w[1,1]] (indexed by k*2+i)
        
        strassen_terms = [
            # (u_local, v_local, w_local) where w_local is indexed by k*2+i
            ([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]),   # M1
            ([0,0, 1,1], [1,0, 0,0], [0,1, 0,-1]),   # M2
            ([1,0, 0,0], [0,1, 0,-1], [0,0, 1,1]),   # M3
            ([0,0, 0,1], [-1,0, 1,0], [1,0, 0,0]),   # M4
            ([1,1, 0,0], [0,0, 0,1], [-1,0, 1,0]),   # M5
            ([-1,0, 1,0], [1,1, 0,0], [0,0, 0,1]),   # M6
            ([0,1, 0,-1], [0,0, 1,1], [1,0, 0,0]),   # M7
        ]
        
        col = base_col
        for u_loc, v_loc, w_loc in strassen_terms:
            U_vals = {}
            V_vals = {}
            W_vals = {}
            
            for li in range(2):
                for lj in range(2):
                    gi = li  # global row (same as local since we're using rows 0,1)
                    gj = offset_j + lj
                    U_vals[gi * m + gj] = u_loc[li * 2 + lj]
            
            for lj in range(2):
                for lk in range(2):
                    gj = offset_j + lj
                    gk = offset_k + lk
                    V_vals[gj * p + gk] = v_loc[lj * 2 + lk]
            
            for lk in range(2):
                for li in range(2):
                    gk = offset_k + lk
                    gi = li
                    W_vals[gk * n + gi] = w_loc[lk * 2 + li]
            
            terms.append((U_vals, V_vals, W_vals, col))
            col += 1
        
        return terms

    # Build the rank-33 decomposition
    # Part 1: Strassen on 2x2 core (cols 0,1 of A) x each 2-col block of B
    # B blocks: cols (0,1), (2,3), (4,5)
    # Part 2: Schoolbook for col 2 of A x all 6 cols of B
    
    U_list = []
    V_list = []
    W_list = []
    col_counter = 0
    
    # Strassen parts: 3 blocks x 7 terms = 21 terms
    for block in range(3):
        offset_k = block * 2  # B column offset
        terms = make_strassen_2x2x2(0, 0, offset_k, col_counter)
        for U_vals, V_vals, W_vals, c in terms:
            U_row = [0] * 33
            V_row = [0] * 33
            W_row = [0] * 33
            for idx, val in U_vals.items():
                U_row[c] = val
            for idx, val in V_vals.items():
                V_row[c] = val
            for idx, val in W_vals.items():
                W_row[c] = val
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)
            col_counter += 1
    
    # Schoolbook tail: A[:,2] x B[:, :] (1x3 x 3x6 = 12 terms)
    for j in range(3):  # row of B (which is col of A, but A only has col 2)
        # Actually: C[i,k] += A[i,2] * B[2,k] for i in {0,1}, k in {0..5}
        # This is 2*6 = 12 terms, but we can do it as 3 terms (one per j) with rank 3*2=6?
        # No: A[i,2]*B[j,k] for j=2 only. So it's 2*6 = 12 separate rank-1 terms.
        # Actually: for each (i,k) pair, C[i,k] += A[i,2]*B[2,k]
        # That's 2*6 = 12 terms where each term has U[i*3+2]=1, V[2*6+k]=1, W[k*2+i]=1
        pass
    
    # Schoolbook: for i in {0,1}, k in {0..5}: term with U[i*3+2]=1, V[2*6+k]=1, W[k*2+i]=1
    for i in range(2):
        for k in range(6):
            U_row = [0] * 33
            V_row = [0] * 33
            W_row = [0] * 33
            U_row[col_counter] = 1
            V_row[col_counter] = 1
            W_row[col_counter] = 1
            # U index: i*3 + 2
            # V index: 2*6 + k
            # W index: k*2 + i
            U_list[i * 3 + 2] = [0] * 33  # placeholder, we'll build properly
            V_list[2 * 6 + k] = [0] * 33
            W_list[k * 2 + i] = [0] * 33
    
    # Let me rebuild this more carefully
    U_mat = [[0] * 33 for _ in range(n * m)]
    V_mat = [[0] * 33 for _ in range(m * p)]
    W_mat = [[0] * 33 for _ in range(p * n)]
    
    col_idx = 0
    # Strassen parts
    for block in range(3):
        offset_k = block * 2
        terms = make_strassen_2x2x2(0, 0, offset_k, col_idx)
        for U_vals, V_vals, W_vals, c in terms:
            for idx, val in U_vals.items():
                U_mat[idx][c] = val
            for idx, val in V_vals.items():
                V_mat[idx][c] = val
            for idx, val in W_vals.items():
                W_mat[idx][c] = val
            col_idx += 1
    
    # Schoolbook tail: A[i,2] * B[2,k] for i in {0,1}, k in {0..5}
    for i in range(2):
        for k in range(6):
            U_mat[i * 3 + 2][col_idx] = 1
            V_mat[2 * 6 + k][col_idx] = 1
            W_mat[k * 2 + i][col_idx] = 1
            col_idx += 1
    
    # Verify
    if verify(U_mat, V_mat, W_mat, 33):
        best_rank = 33
        best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # If the structured approach didn't work, fall back to random search
    if best_rank == 36:
        rng = np.random.RandomState(42)
        for target_rank in range(33, 37):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            max_trials = 20
            for trial in range(max_trials):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                U_init = rng.randn(n * m, target_rank) * 0.5
                V_init = rng.randn(m * p, target_rank) * 0.5
                W_init = rng.randn(p * n, target_rank) * 0.5
                for c in range(target_rank):
                    u_norm = np.linalg.norm(U_init[:, c])
                    v_norm = np.linalg.norm(V_init[:, c])
                    w_norm = np.linalg.norm(W_init[:, c])
                    if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                        scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                        U_init[:, c] /= scale
                        V_init[:, c] /= scale
                        W_init[:, c] /= scale
                U_curr = U_init.copy()
                V_curr = V_init.copy()
                W_curr = W_init.copy()
                lr = 0.05
                for it in range(200):
                    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                        break
                    loss = 0.0
                    gU = np.zeros_like(U_curr)
                    gV = np.zeros_like(V_curr)
                    gW = np.zeros_like(W_curr)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for c in range(target_rank):
                                    total += U_curr[i * m + j, c] * V_curr[j * p + k, c] * W_curr[k * n + i, c]
                                error = 2.0 * (total - 1.0)
                                loss += error * (total - 1.0)
                                for c in range(target_rank):
                                    gU[i * m + j, c] += error * V_curr[j * p + k, c] * W_curr[k * n + i, c]
                                    gV[j * p + k, c] += error * U_curr[i * m + j, c] * W_curr[k * n + i, c]
                                    gW[k * n + i, c] += error * U_curr[i * m + j, c] * V_curr[j * p + k, c]
                    U_curr -= lr * gU
                    V_curr -= lr * gV
                    W_curr -= lr * gW
                    lr *= 0.998
                    if loss < 1e-8:
                        break
                result = try_round_and_verify(U_curr, V_curr, W_curr, target_rank)
                if result is not None:
                    U_r, V_r, W_r = result
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
                    break
    
    return best_result
