import time
import numpy as np
from itertools import product

_START = time.time()
DEADLINE_S = 140.0

def _time_left():
    return DEADLINE_S - (time.time() - _START)

def _strassen_2x2_2():
    """U, V, W (lists) for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_np(Ua, Va, Wa, n, m, p, tol=1e-6):
    """Vectorized verification using numpy."""
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m + j]
            for k in range(p):
                v_row = Va[j*p + k]
                w_row = Wa[k*n + i]
                # sum_c u_c * v_c * w_c
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    return U, V, W

def _try_reduce_als(Ua, Va, Wa, n, m, p, target_rank, time_limit=8.0, seed=0):
    """Try to find a rank-`target_rank` decomposition using alternating least squares.
    Inputs are numpy arrays of the original rank. Returns (U,V,W) as numpy arrays if valid, else None."""
    if _time_left() < time_limit + 2:
        return None
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize: we need to pick target_rank columns from the original decomposition.
    # Strategy: Try to keep the structure as much as possible.
    # The original has 'strassen' columns and 'schoolbook' columns.
    # Simply taking the first target_rank might drop important structure if rank reduction relies on mixing.
    # Better initialization: Random selection of columns? Or keeping the most "important" ones?
    # Let's try a few initialization strategies.
    
    rng = np.random.default_rng(seed)
    
    # Strategy 1: First target_rank columns
    Uo = Ua[:, :target_rank].copy()
    Vo = Va[:, :target_rank].copy()
    Wo = Wa[:, :target_rank].copy()
    
    # Strategy 2: Random columns (maybe better to avoid bias if first columns are not optimal for reduction)
    # We'll stick to Strategy 1 for now but maybe try a perturbation.
    
    # ALS Loop
    for it in range(300):
        if (time.time() - start) > time_limit - 0.2:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((target_rank, p))
                for k in range(p):
                    A[:, k] = Vo[j*p + k] * Wo[k*n + i]
                b = np.ones(p)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Uo[i*m + j] = sol
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for i in range(n):
                    A[:, i] = Uo[i*m + j] * Wo[k*n + i]
                b = np.ones(n)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Vo[j*p + k] = sol
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for j in range(m):
                    A[:, j] = Uo[i*m + j] * Vo[j*p + k]
                b = np.ones(m)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Wo[k*n + i] = sol
        
        # Check error
        # Vectorized check for speed
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = Uo[i*m + j]
                for k in range(p):
                    v_row = Vo[j*p + k]
                    w_row = Wo[k*n + i]
                    val = np.dot(u_row, v_row * w_row)
                    err += (val - 1.0)**2
        if err < 1e-12:
            return Uo, Vo, Wo
            
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def schoolbook():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Deterministic rank-68 construction: 4 disjoint 2x2x2 Strassen blocks
    # Blocks at (i0,j0,k0) = (0,0,0), (0,0,2), (0,1,0), (0,1,2)
    # These cover i in {0,1}, j in {0,1}, k in {0,1,2,3} (Wait, k0+dk where dk in 0,1 so k in 0,1 and 2,3)
    # Covered:
    # Block (0,0,0): i=0,1; j=0,1; k=0,1
    # Block (0,0,2): i=0,1; j=0,1; k=2,3
    # Block (0,1,0): i=0,1; j=1,2; k=0,1  <-- Wait, j0=1, dj in 0,1 -> j=1,2.
    # Block (0,1,2): i=0,1; j=1,2; k=2,3
    
    # Let's check for overlaps.
    # B1: (0,0,0..1), (0,1,0..1), (1,0,0..1), (1,1,0..1)
    # B2: (0,0,2..3), (0,1,2..3), (1,0,2..3), (1,1,2..3)
    # B3: (0,1,0..1), (0,2,0..1), (1,1,0..1), (1,2,0..1)
    # B4: (0,1,2..3), (0,2,2..3), (1,1,2..3), (1,2,2..3)
    
    # Overlaps:
    # (0,1,0..1) is in B1 and B3.
    # (1,1,0..1) is in B1 and B3.
    # (0,1,2..3) is in B2 and B4.
    # (1,1,2..3) is in B2 and B4.
    
    # The current code `_build_strassen_embedding` assumes non-overlapping blocks because it sums contributions.
    # If blocks overlap, the contributions add up. Does the code handle this?
    # The code initializes U, V, W to 0.
    # Then for each block, it adds the Strassen components.
    # If (i,j,k) is covered by multiple blocks, the tensor component T[i,j,k] gets contributions from both.
    # We need T[i,j,k] = 1.
    # If two Strassen blocks contribute to the same (i,j,k), their sum must be 1?
    # Or does the schoolbook part fill in the rest?
    # The code adds schoolbook terms for (i,j,k) NOT in `covered`.
    # `covered` is the union of all block cells.
    # If a cell is in `covered`, it relies solely on the sum of Strassen contributions to equal 1.
    
    # Let's check if the sum of Strassen contributions for overlapping cells equals 1.
    # For a single 2x2x2 Strassen block, the sum of products for any (i,j,k) in that 2x2x2 sub-tensor is 1.
    # If we have two blocks overlapping on some cells, do the contributions add to 1?
    # Not necessarily. Strassen gives 1 for the sub-tensor. If we have two sub-tensors overlapping, the sum is 1 + 1 = 2?
    # If so, the current program's rank 68 solution might be INVALID for overlapping blocks!
    
    # Let's verify the current program's validity.
    strassen_blocks = [(0,0,0), (0,0,2), (0,1,0), (0,1,2)]
    U, V, W = _build_strassen_embedding(n, m, p, strassen_blocks)
    
    # Check validity
    if _verify(U, V, W, n, m, p):
        best_rank = len(U[0])
        best = (U, V, W)
    else:
        # If invalid, fall back to non-overlapping blocks or schoolbook.
        # Let's try to find a valid set of non-overlapping blocks.
        # Total cells 72. Max 2x2x2 blocks = 8 (if 4x4x4 grid). We have 3x3x8.
        # i in 0..2, j in 0..2, k in 0..7.
        # Non-overlapping 2x2x2 blocks:
        # i: 0-1 (1 block) or 1-2 (1 block). Can't have both.
        # j: 0-1 (1 block) or 1-2 (1 block). Can't have both.
        # k: 0-1, 2-3, 4-5, 6-7 (4 blocks).
        # So max non-overlapping blocks is 1 (in i) * 1 (in j) * 4 (in k) = 4 blocks.
        # Or 1 (i) * 2 (j if we use 0-1 and... wait, j=0,1 and j=1,2 overlap at j=1).
        # So max non-overlapping is 4 blocks.
        # Example: i=0,1; j=0,1; k=0,1; 2,3; 4,5; 6,7.
        # Let's try this set.
        strassen_blocks_valid = [(0,0,0), (0,0,2), (0,0,4), (0,0,6)]
        U, V, W = _build_strassen_embedding(n, m, p, strassen_blocks_valid)
        if _verify(U, V, W, n, m, p):
            best_rank = len(U[0])
            best = (U, V, W)
        else:
            # Fallback to schoolbook if even this fails (should not)
            best = schoolbook()
            best_rank = n*m*p
            # Convert to lists
            best = (best["U"], best["V"], best["W"])

    # Now try to reduce rank
    # Convert best to numpy for ALS
    Ua = np.array(best[0], dtype=float)
    Va = np.array(best[1], dtype=float)
    Wa = np.array(best[2], dtype=float)
    
    # Try reducing from best_rank - 1 down to best_rank - 10
    # Use multiple seeds for robustness
    found_reduction = False
    for target in range(best_rank - 1, best_rank - 11, -1):
        if _time_left() < 20.0:
            break
        
        # Try a few seeds
        for seed in range(5):
            if _time_left() < 15.0:
                break
            res = _try_reduce_als(Ua, Va, Wa, n, m, p, target, time_limit=8.0, seed=seed)
            if res is not None:
                rU, rV, rW = res
                # Verify the reduced solution
                if _verify_np(rU, rV, rW, n, m, p):
                    best = (rU.tolist(), rV.tolist(), rW.tolist())
                    best_rank = target
                    found_reduction = True
                    break
        
        if found_reduction:
            # Try to go even lower? The loop continues.
            # But we need to update Ua, Va, Wa to the new best for the next iteration?
            # Actually, ALS reduces from the *original* high rank decomposition.
            # If we found rank R-1, can we reduce R-1 to R-2?
            # ALS might struggle if initialized from the R-1 solution directly.
            # It's safer to always try to reduce from the known valid high-rank decomposition.
            # So we keep Ua, Va, Wa as the original best.
            pass

    # Final safety check
    if _time_left() < 5.0:
        pass

    return {"U": best[0], "V": best[1], "W": best[2]}

DIMENSIONS = [3, 3, 8]
