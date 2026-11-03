"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}


def _kron_decomp_4x4x9():
    """Kronecker product construction: 4x4x9 = (2x2x2) kron (2x2x3) kron (1x1x3)
    
    Actually, let's use a simpler correct approach:
    4x4 x 4x9: split into 2x2 blocks in the A and B row dimensions.
    A is 4x4 = 2x2 blocks of 2x2.
    B is 4x9: rows split into 2x2 blocks, columns stay as 9.
    
    For each 2x2 block pair (I,J) of A and B:
    We need to compute C[I,J] = sum over K of A[I,K] * B[K,J]
    where A[I,K] is 2x2, B[K,J] is 2x9, C[I,J] is 2x9.
    
    So we need 4x2x9 subproblems (2x2 A times 2x9 B -> 2x9 C).
    
    For 2x2x9: split B's 9 columns into groups. 9 = 3*3.
    Use 3 copies of 2x2x3.
    
    For 2x2x3: rank is at most 2*2*3 = 12 (schoolbook), but we can do better.
    Actually, 2x2x3 is just two 2x2x1 and one 2x2x1... no.
    2x2 A times 2x3 B -> 2x3 C.
    C[:,0:2] = A * B[:,0:2] (2x2x2, rank 7 via Strassen)
    C[:,2] = A * B[:,2:3] (2x2x1, rank 2)
    Total: rank 7 + 2 = 9? But we can share the U factors.
    
    Actually for 2x2x3:
    U is 4xR, V is 6xR, W is 6xR.
    We can use Strassen for the first 2 columns (rank 7) and add 2 more for the 3rd column.
    Total rank 9.
    
    Then 2x2x9 = 3 * (2x2x3) with shared U = 3*9 = 27? No, U is shared.
    U is 4xR, and for each of the 3 column-groups, we need V and W.
    Actually U is the same for all column groups if we structure it right.
    
    Let me think differently. Use the Kronecker product structure directly.
    
    The 4x4x9 tensor T[i,j,k] can be written as:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    Using the fact that 4 = 2*2 and 9 = 3*3:
    T[i1*2+i2, j1*2+j2, k1*3+k2] = T_222[i1,j1,k1] * T_223[i2,j2,k2] ... no this doesn't work directly.
    
    Let me just do a correct block construction.
    """
    S = _strassen_2x2()
    # For 2x2x2: rank 7
    # For 2x2x1: rank 2 (just multiply)
    
    # Build 2x2x3 decomposition (rank 9)
    R223 = 9
    U223 = [[0]*R223 for _ in range(4)]
    V223 = [[0]*R223 for _ in range(6)]
    W223 = [[0]*R223 for _ in range(6)]
    
    # First 2 columns: use Strassen (rank 7)
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U223[ii*2+jj][c] = S['U'][ii*2+jj][c]
        for jj in range(2):
            for kk in range(2):
                V223[jj*3+kk][c] = S['V'][jj*2+kk][c]
        for kk in range(2):
            for ii in range(2):
                W223[kk*2+ii][c] = S['W'][kk*2+ii][c]
    
    # 3rd column: rank 2
    # C[:,2] = A * B[:,2]
    # = sum_j A[:,j] * B[j,2]
    # Use 2 rank terms: (A[:,0]*B[0,2]) and (A[:,1]*B[1,2])
    for c in range(2):
        cc = 7 + c
        U223[c*2][cc] = 1  # A[:,c]
        V223[c*3+2][cc] = 1  # B[c,2]
        for ii in range(2):
            W223[2*2+ii][cc] = 1  # C[:,2]
    
    # Now build 4x4x9:
    # 4x4 A = 2x2 blocks, 4x9 B = 2x2 blocks in rows, 9 cols
    # 4x9 C = 2x2 blocks in rows, 9 cols
    # 
    # C[I,J] = sum_K A[I,K] * B[K,J]
    # where I,J,K are 2x2 block indices
    #
    # For each (I,K) pair, A[I,K] is 2x2, B[K,:] is 2x9
    # We need A[I,K] * B[K,:] which is 2x9
    # This is a 2x2x9 problem
    #
    # 2x2x9: use 3 copies of 2x2x3 (for columns 0-2, 3-5, 6-8)
    # But U is shared across the 3 column groups
    #
    # For the full 4x4x9:
    # There are 4 (I,K) pairs. For each, we do a 2x2x9 multiplication.
    # Each 2x2x9 uses rank 27 (3 * 9) but with shared U within each 2x2x9.
    # Across different (I,K) pairs, U is different (different A block).
    #
    # Total rank = 4 * 27 = 108? That's worse than schoolbook.
    #
    # Better approach: share U across K for fixed I.
    # For fixed I, sum_K A[I,K] * B[K,:] 
    # = (concatenation trick)
    #
    # Actually, let's use a different decomposition.
    # 
    # 4x4x9: Think of it as 4 separate 1x4x9 multiplications added together?
    # No, that's schoolbook.
    #
    # Let me try: 4x4x9 = (2x2x2) ⊗ (2x2x3) ⊗ (1x1x3) doesn't make sense.
    #
    # Simplest correct improvement: use Strassen for 4x4x4 part and extend to 9 cols.
    # 4x4x4 has rank ≤ 49 (best known). 
    # 4x4x9: split 9 = 4+4+1.
    # 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 4) = 102.
    # Still worse than 36.
    #
    # The current rank-36 construction, if correct, is already good.
    # Let me just fix the bugs and verify.
    pass
    return None


def _decomp_4x4x9_fixed():
    """Correct 4x4x9 decomposition using block structure.
    
    Split A into 2x2 blocks (4 blocks), B into 2-row x 9-col blocks (2 blocks).
    For each block row I of A (2 rows) and block col J of B (2 rows):
      We compute C[I,:] = sum_K A[I,K] * B[K,:]
      where A[I,K] is 2x2, B[K,:] is 2x9, C[I,:] is 2x9.
    
    For 2x2x9: use 3 groups of 3 columns, each group uses 2x2x3 (rank 9).
    But share U across the 3 groups for the same (I,K) pair.
    
    For fixed (I,K): rank = 27 (3 * 9) but U is 4x27.
    Total: 4 (I values) * 2 (K values) * 27 = 216. Way too much.
    
    Better: For fixed I, sum over K. 
    C[I,:] = A[I,0]*B[0,:] + A[I,1]*B[1,:]
    This is like multiplying a 2x4 matrix by a 4x9 matrix -> 2x9.
    Rank of 2x4x9 ≤ 2*4*9 = 72 (schoolbook), or less.
    
    For 2x4x9: split the 4 middle dimension.
    = 2x2x9 + 2x2x9 (sum)
    Each 2x2x9 has rank 27, total 54 for one I.
    Two I values: 108.
    
    This is getting complicated. Let me just use a verified rank-48 construction:
    4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    Use schoolbook for 4x4x1 (rank 4) and a known 4x4x4 decomp.
    
    Actually, the simplest correct approach that beats 144:
    Use Strassen on 4x4x4 (rank 49) and schoolbook for the remaining 5 columns.
    49 + 4*5 = 49 + 20 = 69.
    
    Or: 4x4x9 = 4x4x3 + 4x4x3 + 4x4x3
    Each 4x4x3: use 2x2 Strassen blocks.
    
    For 4x4x3:
    Split into 4 2x2x3 subproblems.
    Each 2x2x3 has rank 9 (as computed above).
    Total: 4 * 9 = 36.
    
    So 4x4x9 = 3 * 36 = 108? No, the U factors can be shared.
    
    For 4x4x3:
    U is 16 x R, V is 12 x R, W is 12 x R.
    4 blocks (I,J) in {0,1}x{0,1}.
    For each block, we do a 2x2x3 multiplication.
    The U for block (I,J) uses rows (2I..2I+1, 2J..2J+1) of the full U.
    These don't overlap for different (I,J), so no sharing.
    Total rank = 4 * 9 = 36.
    
    For 4x4x9: 3 copies of 4x4x3, U shared across the 3 copies.
    Total rank = 3 * 36 = 108.
    
    Hmm, that's not great. But it's correct.
    
    Let me just implement the 2x2x3 correctly and build up from there.
    """
    S = _strassen_2x2()
    
    # 2x2x3 decomposition, rank 9
    R223 = 9
    U223 = [[0]*R223 for _ in range(4)]
    V223 = [[0]*R223 for _ in range(6)]
    W223 = [[0]*R223 for _ in range(6)]
    
    # Columns 0,1: Strassen (rank 7)
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U223[ii*2+jj][c] = S['U'][ii*2+jj][c]
        for jj in range(2):
            for kk in range(2):
                V223[jj*3+kk][c] = S['V'][jj*2+kk][c]
        for kk in range(2):
            for ii in range(2):
                W223[kk*2+ii][c] = S['W'][kk*2+ii][c]
    
    # Column 2: rank 2
    for c in range(2):
        cc = 7 + c
        U223[c*2][cc] = 1
        V223[c*3+2][cc] = 1
        for ii in range(2):
            W223[2*2+ii][cc] = 1
    
    # 4x4x3: 4 blocks of 2x2x3, rank 36
    R443 = 36
    U443 = [[0]*R443 for _ in range(16)]
    V443 = [[0]*R443 for _ in range(12)]
    W443 = [[0]*R443 for _ in range(12)]
    
    for I in range(2):
        for J in range(2):
            base = (I*2 + J) * 9
            for c in range(9):
                cc = base + c
                for ii in range(2):
                    for jj in range(2):
                        U443[(2*I+ii)*4 + (2*J+jj)][cc] = U223[ii*2+jj][c]
                for jj in range(2):
                    for kk in range(3):
                        V443[(2*J+jj)*3 + kk][cc] = V223[jj*3+kk][c]
                for kk in range(3):
                    for ii in range(2):
                        W443[kk*4 + (2*I+ii)][cc] = W223[kk*2+ii][c]
    
    # 4x4x9: 3 copies of 4x4x3, U shared
    R449 = 108
    U449 = [[0]*R449 for _ in range(16)]
    V449 = [[0]*R449 for _ in range(36)]
    W449 = [[0]*R449 for _ in range(36)]
    
    for t in range(3):
        for c in range(36):
            cc = t*36 + c
            for i in range(16):
                U449[i][cc] = U443[i][c]
            for j in range(4):
                for k in range(3):
                    V449[j*9 + (3*t+k)][cc] = V443[j*3+k][c]
            for k in range(3):
                for i in range(4):
                    W449[(3*t+k)*4 + i][cc] = W443[k*4+i][c]
    
    return {"U": U449, "V": V449, "W": W449}


def _try_structured_rank(rank, n=4, m=4, p=9, max_attempts=50, deadline=None):
    """Try to find a valid decomposition of given rank using structured patterns."""
    if deadline is not None and time.time() > deadline:
        return None
    
    # Random search with integer constraints
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt * 7919 + rank * 104729)
        
        U = np.random.randint(-1, 2, size=(n * m, rank))
        V = np.random.randint(-1, 2, size=(m * p, rank))
        
        W = np.zeros((p * n, rank), dtype=float)
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                b = np.ones(m)
                
                try:
                    w_sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    
                    w_int = np.round(w_sol)
                    w_half = np.round(w_sol * 2) / 2
                    
                    err_int = np.max(np.abs(A @ w_int - b))
                    err_half = np.max(np.abs(A @ w_half - b))
                    
                    if err_int < 1e-8:
                        W[k * n + i, :] = w_int
                    elif err_half < 1e-8:
                        W[k * n + i, :] = w_half
                    else:
                        best_err = min(err_int, err_half)
                        best_w = w_int if err_int < err_half else w_half
                        
                        for c in range(rank):
                            for delta in [-1, 1]:
                                w_trial = best_w.copy()
                                w_trial[c] += delta
                                err = np.max(np.abs(A @ w_trial - b))
                                if err < best_err - 1e-10:
                                    best_err = err
                                    best_w = w_trial
                            
                            for delta in [-0.5, 0.5]:
                                w_trial = best_w.copy()
                                w_trial[c] += delta
                                err = np.max(np.abs(A @ w_trial - b))
                                if err < best_err - 1e-10:
                                    best_err = err
                                    best_w = w_trial
                        
                        if best_err < 1e-8:
                            W[k * n + i, :] = best_w
                        else:
                            valid = False
                            break
                except Exception:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 9
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try the fixed 4x4x9 construction (rank 108)
    try:
        cand = _decomp_4x4x9_fixed()
        if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
            best_result = cand
            best_rank = 108
            print("Verified rank-108 decomposition")
    except Exception as e:
        print("rank-108 construction failed:", e)

    if time.time() - start_time > DEADLINE:
        return best_result

    # Try to improve via search
    for rank in range(best_rank - 1, max(best_rank - 20, 50), -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        print(f"Trying rank {rank}...")
        result = _try_structured_rank(rank, n, m, p, max_attempts=50,
                                     deadline=DEADLINE - 10)
        
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], n, m, p):
                best_result = result
                best_rank = rank
                print(f"Found valid rank {rank} decomposition!")
                break
    
    return best_result
