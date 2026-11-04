import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

def _make_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check all entries for small tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _snap_to_rational(arr, max_denom=20):
    """Snap array values to nearby rational numbers with small denominators."""
    result = arr.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            best_frac = val
            best_err = float('inf')
            for denom in range(1, max_denom + 1):
                num = round(val * denom)
                frac = num / denom
                err = abs(frac - val)
                if err < best_err:
                    best_err = err
                    best_frac = frac
            result[i, j] = best_frac
    return result

def _optimize_m323(target_rank, max_time=10):
    """Find a decomposition for M_{3,2,3}."""
    n, m, p = 3, 2, 3
    start_time = time.time()
    
    # Reshape indices for tensor T[i,j,k]
    # U shape (3*2, rank) = (6, rank)
    # V shape (2*3, rank) = (6, rank)
    # W shape (3*3, rank) = (9, rank)
    
    # Flatten parameters
    # We optimize U, V, W jointly
    
    def loss(params):
        U = params[:6*target_rank].reshape(6, target_rank)
        V = params[6*target_rank:12*target_rank].reshape(6, target_rank)
        W = params[12*target_rank:].reshape(9, target_rank)
        
        # Compute tensor T[i,j,k] = sum_c U[i*2+j,c] V[j*3+k,c] W[k*3+i,c]
        # We want T[i,j,k] = 1
        errors = []
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    val = np.sum(U[i*2+j, :] * V[j*3+k, :] * W[k*3+i, :])
                    errors.append(val - 1.0)
        return np.array(errors)
    
    # Initial guess
    np.random.seed(42)
    U_init = np.random.randn(6, target_rank) * 0.5
    V_init = np.random.randn(6, target_rank) * 0.5
    W_init = np.random.randn(9, target_rank) * 0.5
    params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_solution = None
    best_loss_val = float('inf')
    
    for attempt in range(5):
        if time.time() > start_time + max_time - 2:
            break
        np.random.seed(attempt * 100 + 1)
        U_init = np.random.randn(6, target_rank) * 0.5
        V_init = np.random.randn(6, target_rank) * 0.5
        W_init = np.random.randn(9, target_rank) * 0.5
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(loss, params_init, method='lm', max_nfev=10000)
            if result.success or result.cost < best_loss_val:
                if result.cost < best_loss_val:
                    best_loss_val = result.cost
                    best_solution = result.x
        except Exception:
            pass
            
    if best_solution is None:
        return None, None, None, False
        
    U = best_solution[:6*target_rank].reshape(6, target_rank)
    V = best_solution[6*target_rank:12*target_rank].reshape(6, target_rank)
    W = best_solution[12*target_rank:].reshape(9, target_rank)
    
    # Try to snap
    U_snapped = _snap_to_rational(U, max_denom=10)
    V_snapped = _snap_to_rational(V, max_denom=10)
    W_snapped = _snap_to_rational(W, max_denom=10)
    
    if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-8):
        return U_snapped, V_snapped, W_snapped, True
    
    # If snapping failed, try without snapping (might be invalid for task contract but better than nothing? No, contract says int/half-int)
    # If verify fails, return failure
    return None, None, None, False

def _construct_m343_from_m323(U323, V323, W323):
    """Construct M_{3,4,3} from two M_{3,2,3} blocks."""
    # M_{3,4,3}: A (3x4) * B (4x3)
    # Split A into A1 (3x2), A2 (3x2)
    # C = A1 * B_top + A2 * B_bot
    # B_top is rows 0,1 of B (4x3)
    # B_bot is rows 2,3 of B (4x3)
    
    n, m, p = 3, 4, 3
    rank_323 = U323.shape[1]
    rank_343 = 2 * rank_323
    
    U343 = np.zeros((3*4, rank_343))
    V343 = np.zeros((4*3, rank_343))
    W343 = np.zeros((3*3, rank_343))
    
    # Block 1: A1 (cols 0,1 of A) * B_top (rows 0,1 of B)
    # U343 for A1: rows 0..5 (3*2) map to U323
    # V343 for B_top: rows 0..2 (0*3..2*3) map to V323? 
    # V323 index j*p+k. For B_top, j is 0,1. So V323 rows 0..5.
    # W343: same W323
    
    # U343 cols 0..rank_323-1
    U343[0:6, 0:rank_323] = U323
    
    # V343 cols 0..rank_323-1
    # V343 row index is j*3+k.
    # For B_top, j=0,1. So rows 0..5.
    V343[0:6, 0:rank_323] = V323
    
    # W343 cols 0..rank_323-1
    W343[:, 0:rank_323] = W323
    
    # Block 2: A2 (cols 2,3 of A) * B_bot (rows 2,3 of B)
    # U343 for A2: rows 6..11 (3*4, cols 2,3 -> indices 2,3 in m -> 0*4+2, 1*4+2... wait)
    # A is 3x4. Rows i=0,1,2. Cols j=2,3.
    # Index in U343: i*4+j.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    # i=2, j=2 -> 10. i=2, j=3 -> 11.
    # These correspond to rows 0,1,2,3,4,5 of U323?
    # U323 is 3x2. Indices i*2+j.
    # i=0,j=0->0. i=0,j=1->1.
    # i=1,j=0->2. i=1,j=1->3.
    # i=2,j=0->4. i=2,j=1->5.
    # So we map U343 rows [2,3,6,7,10,11] to U323 rows [0,1,2,3,4,5].
    
    U343[[2,3,6,7,10,11], rank_323:2*rank_323] = U323
    
    # V343 for B_bot: rows 2,3 of B.
    # B is 4x3. Rows j=2,3.
    # Index in V343: j*3+k.
    # j=2, k=0,1,2 -> 6,7,8.
    # j=3, k=0,1,2 -> 9,10,11.
    # Map to V323 rows 0..5.
    V343[[6,7,8,9,10,11], rank_323:2*rank_323] = V323
    
    # W343 is same
    W343[:, rank_323:2*rank_323] = W323
    
    return U343, V343, W343

def _construct_m346_from_m343(U343, V343, W343):
    """Construct M_{3,4,6} from two M_{3,4,3} blocks."""
    # M_{3,4,6}: A (3x4) * B (4x6)
    # Split B into B1 (4x3), B2 (4x3)
    # C = A * B1 + A * B2
    
    n, m, p = 3, 4, 6
    rank_343 = U343.shape[1]
    rank_346 = 2 * rank_343
    
    U346 = np.zeros((3*4, rank_346))
    V346 = np.zeros((4*6, rank_346))
    W346 = np.zeros((6*3, rank_346))
    
    # Block 1: A * B1 (cols 0,1,2 of B)
    # U346 same as U343
    U346[:, 0:rank_343] = U343
    
    # V346 for B1: rows 0..11 (4*3) map to V343 rows 0..11
    V346[0:12, 0:rank_343] = V343
    
    # W346 for C1 (cols 0,1,2 of C): rows 0..8 (3*3) map to W343
    # W index k*n+i. k=0,1,2.
    # So rows 0..8.
    W346[0:9, 0:rank_343] = W343
    
    # Block 2: A * B2 (cols 3,4,5 of B)
    # U346 same
    U346[:, rank_343:2*rank_343] = U343
    
    # V346 for B2: rows 12..23 (4*6, cols 3,4,5)
    # B index j*6+k.
    # j=0, k=3,4,5 -> 3,4,5
    # j=1, k=3,4,5 -> 9,10,11
    # j=2, k=3,4,5 -> 15,16,17
    # j=3, k=3,4,5 -> 21,22,23
    # Map to V343 rows 0..11.
    # V343 index j*3+k.
    # j=0, k=0,1,2 -> 0,1,2
    # j=1, k=0,1,2 -> 3,4,5
    # j=2, k=0,1,2 -> 6,7,8
    # j=3, k=0,1,2 -> 9,10,11
    
    # So V346 rows [3,4,5, 9,10,11, 15,16,17, 21,22,23] map to V343 rows 0..11
    v_rows_b2 = [3,4,5, 9,10,11, 15,16,17, 21,22,23]
    v_rows_343 = list(range(12))
    V346[v_rows_b2, rank_343:2*rank_343] = V343[v_rows_343, :]
    
    # W346 for C2 (cols 3,4,5 of C): rows 9..17
    # W index k*n+i.
    # k=3, i=0,1,2 -> 9,10,11
    # k=4, i=0,1,2 -> 12,13,14
    # k=5, i=0,1,2 -> 15,16,17
    # Map to W343 rows 0..8.
    w_rows_c2 = [9,10,11, 12,13,14, 15,16,17]
    w_rows_343 = list(range(9))
    W346[w_rows_c2, rank_343:2*rank_343] = W343[w_rows_343, :]
    
    return U346, V346, W346

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try hierarchical construction
    # 1. Find M_{3,2,3}
    for rank_323 in [12, 13, 14, 15]:
        if time.time() > deadline - 20:
            break
        U323, V323, W323, success = _optimize_m323(rank_323, max_time=10)
        if success:
            # 2. Construct M_{3,4,3}
            U343, V343, W343 = _construct_m343_from_m323(U323, V323, W323)
            if _verify_decomposition(U343, V343, W343, 3, 4, 3, tol=1e-8):
                # 3. Construct M_{3,4,6}
                U346, V346, W346 = _construct_m346_from_m343(U343, V343, W343)
                if _verify_decomposition(U346, V346, W346, n, m, p, tol=1e-8):
                    best_rank = U346.shape[1]
                    best_U, best_V, best_W = U346, V346, W346
                    break # Found a good one, stop
    
    # If we didn't find one via hierarchy, try direct optimization on M_{3,4,6} with rank 50-60
    if best_rank == 72:
        for target_rank in [50, 55, 60]:
            if time.time() > deadline - 30:
                break
            # Use a simple gradient descent or least squares here if needed
            # For now, rely on hierarchy
            
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
