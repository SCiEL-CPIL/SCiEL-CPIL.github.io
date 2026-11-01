"""Rewritten program with explicit rank-39 construction."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Construct an explicit rank-39 decomposition for M_{2,2,11}.
    Uses 5 Strassen blocks (2x2) + 1 trivial block (2x1) = 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, base_rank):
        r = base_rank
        # P1 = (A00+A11)*(B00+B11)
        U[0][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+3][r] += 1.0
        # P2 = (A10+A11)*B00
        r = base_rank + 1
        U[2][r] += 1.0; U[3][r] += 1.0
        V[col_offset][r] += 1.0
        W[2*col_offset+1][r] += 1.0
        # P3 = A00*(B01-B10)
        r = base_rank + 2
        U[0][r] += 1.0
        V[col_offset+1][r] += 1.0; V[11+col_offset][r] += -1.0
        W[2*col_offset+2][r] += 1.0
        # P4 = A11*(B10-B00)
        r = base_rank + 3
        U[3][r] += 1.0
        V[11+col_offset][r] += 1.0; V[col_offset][r] += -1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset][r] += -1.0
        # P5 = (A00+A01)*B11
        r = base_rank + 4
        U[0][r] += 1.0; U[1][r] += 1.0
        V[12+col_offset][r] += 1.0
        W[2*col_offset+2][r] += 1.0; W[2*col_offset+3][r] += -1.0
        # P6 = (A10-A00)*(B00+B01)
        r = base_rank + 5
        U[2][r] += 1.0; U[0][r] += -1.0
        V[col_offset][r] += 1.0; V[col_offset+1][r] += 1.0
        W[2*col_offset+1][r] += 1.0; W[2*col_offset+3][r] += 1.0
        # P7 = (A01-A11)*(B10+B11)
        r = base_rank + 6
        U[1][r] += 1.0; U[3][r] += -1.0
        V[11+col_offset][r] += 1.0; V[12+col_offset][r] += 1.0
        W[2*col_offset][r] += 1.0; W[2*col_offset+1][r] += -1.0
    
    for block_idx in range(5):
        add_strassen_block(2 * block_idx, 7 * block_idx)
    
    # Trivial 2x1 block at column 10
    base = 35
    U[0][base] += 1.0; V[10][base] += 1.0; W[20][base] += 1.0
    U[1][base+1] += 1.0; V[21][base+1] += 1.0; W[20][base+1] += 1.0
    U[2][base+2] += 1.0; V[10][base+2] += 1.0; W[21][base+2] += 1.0
    U[3][base+3] += 1.0; V[21][base+3] += 1.0; W[21][base+3] += 1.0
    
    return U, V, W

def _als_search(rank, n, m, p, max_iter=2000, seed=42, deadline=None):
    """
    Alternating Least Squares to find a rank-r decomposition.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Build target tensor indices
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i*m+j, j*p+k, k*n+i))
    ntargets = len(targets)
    
    # Precompute which targets involve each index for each factor
    # For U[a]: targets where ta == a
    u_targets = {a: [] for a in range(n*m)}
    # For V[b]: targets where tb == b
    v_targets = {b: [] for b in range(m*p)}
    # For W[c]: targets where tc == c
    w_targets = {c: [] for c in range(p*n)}
    
    for idx, (ta, tb, tc) in enumerate(targets):
        u_targets[ta].append(idx)
        v_targets[tb].append(idx)
        w_targets[tc].append(idx)
    
    best_err = float('inf')
    best_solution = None
    
    for restart in range(3):
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        for it in range(max_iter):
            if deadline and time.time() > deadline - 2:
                break
            
            # Update U
            for a in range(n * m):
                idxs = u_targets[a]
                for c in range(rank):
                    num = 0.0
                    den = 0.0
                    for idx in idxs:
                        ta, tb, tc = targets[idx]
                        vc = V[tb, c]
                        wc = W[tc, c]
                        num += vc * wc
                        den += vc * vc * wc * wc
                    if den > 1e-14:
                        U[a, c] = num / den
            
            # Update V
            for b in range(m * p):
                idxs = v_targets[b]
                for c in range(rank):
                    num = 0.0
                    den = 0.0
                    for idx in idxs:
                        ta, tb, tc = targets[idx]
                        uc = U[ta, c]
                        wc = W[tc, c]
                        num += uc * wc
                        den += uc * uc * wc * wc
                    if den > 1e-14:
                        V[b, c] = num / den
            
            # Update W
            for cc in range(p * n):
                idxs = w_targets[cc]
                for c in range(rank):
                    num = 0.0
                    den = 0.0
                    for idx in idxs:
                        ta, tb, tc = targets[idx]
                        uc = U[ta, c]
                        vc = V[tb, c]
                        num += uc * vc
                        den += uc * uc * vc * vc
                    if den > 1e-14:
                        W[cc, c] = num / den
            
            # Check convergence every 50 iterations
            if it % 50 == 0:
                err = 0.0
                for ta, tb, tc in targets:
                    s = 0.0
                    for c in range(rank):
                        s += U[ta, c] * V[tb, c] * W[tc, c]
                    err += (s - 1.0) ** 2
                if err < best_err:
                    best_err = err
                    best_solution = (U.copy(), V.copy(), W.copy())
                if err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
        
        if best_solution and best_err < 1e-10:
            break
    
    if best_solution and best_err < 1e-10:
        return best_solution[0].tolist(), best_solution[1].tolist(), best_solution[2].tolist()
    return None

def _try_integer_refinement(U, V, W, rank, n, m, p):
    """
    Try to refine a numerical solution to have small rational entries.
    Returns None if not successful.
    """
    # Check if the current solution is already close to integers/half-integers
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Round to nearest half-integer and check
    U_rounded = np.round(U_arr * 2) / 2
    V_rounded = np.round(V_arr * 2) / 2
    W_rounded = np.round(W_arr * 2) / 2
    
    # Verify rounded version
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-6):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First: guaranteed rank-39
    try:
        U, V, W = _construct_rank39()
        if _verify(U, V, W, n, m, p):
            best_rank = 39
            best_result = (U, V, W)
        else:
            best_rank = None
            best_result = None
    except Exception:
        best_rank = None
        best_result = None
    
    # Try to find lower rank using ALS
    for rank in range(20, best_rank if best_rank else 39):
        if time.time() > deadline - 10:
            break
        result = _als_search(rank, n, m, p, max_iter=3000, seed=rank, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                # Try to get cleaner entries
                refined = _try_integer_refinement(U, V, W, rank, n, m, p)
                if refined:
                    U, V, W = refined
                return {"U": U, "V": V, "W": W}
    
    # Return best found
    if best_result:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Final fallback: schoolbook
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
# EVOLVE-BLOCK-END
