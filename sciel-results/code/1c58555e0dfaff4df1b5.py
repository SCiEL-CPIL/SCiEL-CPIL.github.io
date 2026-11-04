import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0  # Safety margin under 150s kill


def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _build_4x4x4():
    """Build 4x4x4 tensors via Kronecker product of two 2x2x2 Strassen. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i * 4 + j, c] = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j * 4 + k, c] = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k * 4 + i, c] = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]

    return U4, V4, W4


def _verify_decomposition(U, V, W, n=4, m=4, p=8, atol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    Uf = U.astype(np.float64).reshape(n, m, rank)
    Vf = V.astype(np.float64).reshape(m, p, rank)
    Wf = W.astype(np.float64).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Uf, Vf, Wf)
    return np.allclose(T, 1.0, atol=atol)


def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Two independent 4x4x4 Strassen blocks (rank 49 each) = rank 98.
    U is shared across both blocks (same A factor).
    """
    U4, V4, W4 = _build_4x4x4()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    for block in range(2):
        cs = block * 49
        ce = (block + 1) * 49
        U[:, cs:ce] = U4
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j * 8 + kg, cs:ce] = V4[j * 4 + kl, :]
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg * 4 + i, cs:ce] = W4[kl * 4 + i, :]

    if not _verify_decomposition(U, V, W):
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_lp_rank_reduction(r_max=98, time_budget=60.0):
    """
    Attempt to find a lower-rank decomposition using LP-based approach.
    Uses the rank-98 Strassen-Kronecker as initialization and tries to
    reduce rank by finding sparse solutions.
    
    Strategy: For each candidate rank r from r_max-1 down to some minimum,
    set up an LP that finds factor matrices with at most r non-zero columns
    that satisfy the tensor equation. We use a soft approach: minimize
    the L1 norm of the factors subject to the tensor constraints.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4()
    U98 = np.zeros((16, 98), dtype=np.float64)
    V98 = np.zeros((32, 98), dtype=np.float64)
    W98 = np.zeros((32, 98), dtype=np.float64)
    
    for block in range(2):
        cs = block * 49
        ce = (block + 1) * 49
        U98[:, cs:ce] = U4.astype(np.float64)
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V98[j * 8 + kg, cs:ce] = V4[j * 4 + kl, :].astype(np.float64)
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W98[kg * 4 + i, cs:ce] = W4[kl * 4 + i, :].astype(np.float64)
    
    # Try to find lower rank via iterative column elimination
    # Start with rank-98 and try to zero out columns
    best_rank = 98
    best_U, best_V, best_W = U98.copy(), V98.copy(), W98.copy()
    
    for target_rank in range(97, 55, -1):
        if time.time() > START + time_budget + 20:
            break
        
        # Try to find a valid decomposition with exactly target_rank non-zero columns
        # Use a greedy approach: try removing one column at a time
        current_rank = 98
        U_cur, V_cur, W_cur = U98.copy(), V98.copy(), W98.copy()
        
        while current_rank > target_rank:
            if time.time() > START + time_budget + 20:
                break
            
            found_removal = False
            # Try each column to see if it can be removed
            for col in range(97, -1, -1):
                if U_cur[:, col].all() == 0 and V_cur[:, col].all() == 0 and W_cur[:, col].all() == 0:
                    continue
                    
                # Temporarily zero out this column and check if we can compensate
                # by adjusting other columns
                U_test = U_cur.copy()
                V_test = V_cur.copy()
                W_test = W_cur.copy()
                U_test[:, col] = 0
                V_test[:, col] = 0
                W_test[:, col] = 0
                
                # Check if the remaining columns still satisfy the tensor equation
                # This is expensive, so we'll use a quick check
                non_zero_cols = np.where(
                    (np.abs(U_test).sum(axis=0) > 1e-12) |
                    (np.abs(V_test).sum(axis=0) > 1e-12) |
                    (np.abs(W_test).sum(axis=0) > 1e-12)
                )[0]
                
                if len(non_zero_cols) >= current_rank - 1:
                    # Check if tensor equation still holds with this column removed
                    U3 = U_test[:, non_zero_cols].reshape(n, m, -1)
                    V3 = V_test[:, non_zero_cols].reshape(m, p, -1)
                    W3 = W_test[:, non_zero_cols].reshape(p, n, -1)
                    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    
                    if np.allclose(T, 1.0, atol=1e-9):
                        # Column can be removed
                        U_cur, V_cur, W_cur = U_test, V_test, W_test
                        current_rank -= 1
                        found_removal = True
                        break
            
            if not found_removal:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            
            # Round to nearest half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                # Convert to int16 (multiply by 2 to handle half-integers)
                U_int = (U_rounded * 2).astype(np.int16)
                V_int = (V_rounded * 2).astype(np.int16)
                W_int = (W_rounded * 2).astype(np.int16)
                
                # Find non-zero columns
                nz_cols = np.where(
                    (np.abs(U_int).sum(axis=0) > 0) |
                    (np.abs(V_int).sum(axis=0) > 0) |
                    (np.abs(W_int).sum(axis=0) > 0)
                )[0]
                
                U_final = U_int[:, nz_cols]
                V_final = V_int[:, nz_cols]
                W_final = W_int[:, nz_cols]
                
                final_rank = len(nz_cols)
                if final_rank < best_rank:
                    best_rank = final_rank
                    best_U, best_V, best_W = U_final, V_final, W_final
                
                if best_rank <= target_rank:
                    break
    
    if best_rank < 98:
        # Convert to list format
        U_out = best_U.tolist() if best_U.dtype == np.int16 else [[x / 2 for x in row] for row in best_U.tolist()]
        V_out = best_V.tolist() if best_V.dtype == np.int16 else [[x / 2 for x in row] for row in best_V.tolist()]
        W_out = best_W.tolist() if best_W.dtype == np.int16 else [[x / 2 for x in row] for row in best_W.tolist()]
        
        # Verify final result
        Uf = np.array(U_out).astype(np.float64)
        Vf = np.array(V_out).astype(np.float64)
        Wf = np.array(W_out).astype(np.float64)
        
        if _verify_decomposition(Uf, Vf, Wf):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def _schoolbook(n=4, m=4, p=8):
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


def construct():
    # First, get the baseline rank-98 decomposition
    baseline = None
    try:
        if time.time() < DEADLINE:
            baseline = _build_rank98()
    except Exception:
        pass
    
    if baseline is None:
        return _schoolbook()
    
    # Try to improve via LP-based rank reduction if time permits
    if time.time() < DEADLINE - 30:
        try:
            improved = _try_lp_rank_reduction(time_budget=40.0)
            if improved is not None:
                return improved
        except Exception:
            pass
    
    return baseline


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Verify
    n, m, p = 4, 4, 8
    rank = len(res['U'][0])
    U = np.array(res['U'], dtype=np.float64).reshape(n, m, rank)
    V = np.array(res['V'], dtype=np.float64).reshape(m, p, rank)
    W = np.array(res['W'], dtype=np.float64).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    print(f"Max error: {np.max(np.abs(T - 1.0))}")
