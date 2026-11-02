import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135


def _build_rank_32(n, m, p):
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using einsum."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by removing terms."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_removal(indices_to_remove):
        mask = [i for i in range(rank) if i not in indices_to_remove]
        if len(mask) == 0:
            return None
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = len(mask)
            return (
                [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
            )
        return None

    # Try removing each single term
    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        result = check_removal([r])
        if result is not None:
            return result

    # Try removing pairs
    for r1 in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        for r2 in range(r1 + 1, rank):
            result = check_removal([r1, r2])
            if result is not None:
                return result

    # Try removing triples if we have time
    if time.time() - _START_TIME < _TIMEOUT - 15:
        for r1 in range(rank):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            for r2 in range(r1 + 1, rank):
                if time.time() - _START_TIME > _TIMEOUT - 5:
                    return None
                for r3 in range(r2 + 1, rank):
                    if time.time() - _START_TIME > _TIMEOUT - 5:
                        return None
                    result = check_removal([r1, r2, r3])
                    if result is not None:
                        return result

    return None


def _try_rank_28_cyclic(n, m, p):
    """
    Try to find a rank-28 decomposition by exploiting cyclic structure.
    
    Idea: The 2x2x9 tensor can potentially be decomposed more efficiently
    than 4*Strassen + 1*trivial = 32.
    
    Strategy: Try different groupings of the 9 columns into blocks and
    use optimized decompositions for each block size.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Strategy 1: Try 3 Strassen blocks (rank 7 each) + 1 block of 3 columns
    # For 3 columns, we might need rank 10 or less
    # Total: 3*7 + rank_3 = 21 + rank_3. Need rank_3 <= 7 for total <= 28.
    # But 3 columns trivially need rank 12. Let's see if we can do better.
    
    # Actually, let's try a different approach: use the fact that the tensor
    # is a direct sum of 9 rank-4 tensors (one per column k).
    # T = sum_k T_k where T_k has non-zero entries only at (i,j),(j,k),(k,i)
    
    # Each T_k is a rank-4 tensor (identity-like). The question is whether
    # we can share factors between different T_k's.
    
    # Let's try: group columns into {0,1,2,3,4,5,6,7,8} and try various
    # block decompositions.
    
    # Block decomposition options:
    # Option A: [7,1] -> 7 + 4 = 11 per block, 8 blocks... no
    # Option B: [4,4,1] -> Strassen(4) + Strassen(4) + trivial(1)
    #   Strassen for 2x2x4: can we do better than 14?
    #   Actually 2x2x2 is rank 7, 2x2x4 = 2 * 2x2x2, so rank 14.
    #   So [4,4,1] -> 14 + 14 + 4 = 32. Same as before.
    
    # Option C: Try to find a non-trivial decomposition of 2x2x3
    # 2x2x3 tensor: 12 non-zero entries. Lower bound is ceil(12/4) = 3? No.
    # The tensor rank of 2x2x3 is at most 12 (trivial), at least...
    # Let's just try to solve it numerically.
    
    # For now, let's try a more systematic search:
    # Build a rank-32 decomposition and try to find linear dependencies
    # in the factor matrices that allow rank reduction.
    
    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)
    
    # Check if any columns of U, V, W are linearly dependent in a way
    # that allows rank reduction.
    # Specifically, check if the decomposition has redundant terms.
    
    # For each pair of terms (r1, r2), check if removing both and adjusting
    # other terms still gives the correct tensor.
    # This is equivalent to checking if the sum of two rank-1 tensors can be
    # replaced by a lower-rank decomposition of their sum.
    
    # More practically: check if any single term is redundant (rank 31)
    # or any two terms are redundant (rank 30), etc.
    
    # We already do this in _try_reduce_rank, but let's also try a different
    # approach: modify the decomposition to create dependencies.
    
    # Alternative: try a different initial decomposition that might have
    # more structure.
    
    # Let's try: use 3 Strassen blocks on columns {0,1}, {2,3}, {4,5}
    # and handle columns {6,7,8} with a custom rank-11 decomposition
    # (if it exists) for 2x2x3.
    
    # First, let's see if 2x2x3 has rank < 12.
    # The 2x2x3 tensor T3 has entries T3[i*m+j, j*3+k, k*2+i] = 1
    # for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # That's 12 entries.
    
    # Let's try to find a rank-11 decomposition of 2x2x3 by solving
    # a system of equations.
    
    def _find_rank_r_decomposition_2x2x3(r, max_time=30):
        """Try to find a rank-r decomposition of the 2x2x3 tensor."""
        start = time.time()
        if time.time() - _START_TIME > _TIMEOUT - 15:
            return None
        
        # The 2x2x3 tensor has 12 non-zero entries.
        # A rank-r decomposition has 4*r + 6*r + 6*r = 16*r parameters
        # (with 4, 6, 6 entries in U, V, W respectively).
        # We need to satisfy 12 equations (one per non-zero entry) plus
        # ensure all other entries are zero.
        
        # Total entries in the tensor: 4 * 6 * 6 = 144
        # Non-zero: 12, zero: 132
        # So we have 144 equations and 16*r unknowns.
        # For r=11: 176 unknowns, 144 equations. Should be solvable.
        
        # Use a simple approach: set up the equations and solve.
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        
        # Let's use a random initialization and gradient descent.
        np.random.seed(42)
        
        for attempt in range(10):
            U = np.random.randn(4, r) * 0.1
            V = np.random.randn(6, r) * 0.1
            W = np.random.randn(6, r) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() - _START_TIME > _TIMEOUT - 10:
                    return None
                if time.time() - start > max_time:
                    break
                
                # Compute T_recon
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                
                # Compute error
                error = np.zeros((4, 6, 6))
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            error[i*2+j, j*3+k, k*2+i] = 1.0
                error -= T_recon
                
                if np.linalg.norm(error) < 1e-10:
                    return U, V, W
                
                # Compute gradients
                # dT/dU[i,r] = V[j,r] * W[k,r] for each (i,j,k)
                # d(error^2)/dU[i,r] = 2 * sum_{j,k} error[i,j,k] * V[j,r] * W[k,r]
                
                dU = 2 * np.einsum('ijk,jr,kr->ir', error, V, W)
                dV = 2 * np.einsum('ijk,ir,kr->jr', error, U, W)
                dW = 2 * np.einsum('ijk,ir,jr->kr', error, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Normalize to prevent explosion
                if np.linalg.norm(U) > 100:
                    U /= np.linalg.norm(U)
                if np.linalg.norm(V) > 100:
                    V /= np.linalg.norm(V)
                if np.linalg.norm(W) > 100:
                    W /= np.linalg.norm(W)
        
        return None
    
    # Try to find rank-11 decomposition of 2x2x3
    result_2x2x3 = _find_rank_r_decomposition_2x2x3(11)
    if result_2x2x3 is not None:
        U3, V3, W3 = result_2x2x3
        # Verify
        T3 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T3[i*2+j, j*3+k, k*2+i] = 1.0
        T3_recon = np.einsum('ir,jr,kr->ijk', U3, V3, W3)
        if np.allclose(T3, T3_recon, atol=1e-8):
            # Great! Now build the full decomposition:
            # 3 Strassen blocks (rank 7 each) + 1 custom block (rank 11)
            # Total: 3*7 + 11 = 32. Still 32, not 28.
            # Need rank-8 for 2x2x3 to get total 28.
            pass
    
    # Try rank-8 for 2x2x3 (would give total 3*7+8 = 29, still not 28)
    # Try rank-7 for 2x2x3 (would give total 3*7+7 = 28!)
    result_2x2x3_r7 = _find_rank_r_decomposition_2x2x3(7)
    if result_2x2x3_r7 is not None:
        U3, V3, W3 = result_2x2x3_r7
        T3 = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T3[i*2+j, j*3+k, k*2+i] = 1.0
        T3_recon = np.einsum('ir,jr,kr->ijk', U3, V3, W3)
        if np.allclose(T3, T3_recon, atol=1e-8):
            # Build full rank-28 decomposition
            rank = 28
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(18)]
            W = [[0] * rank for _ in range(18)]
            
            col = 0
            # Add 3 Strassen blocks for columns {0,1}, {2,3}, {4,5}
            for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
                # M1: (A00+A11)*(B00+B11)
                U[0][col] = 1
                U[3][col] = 1
                V[0*p + k0][col] = 1
                V[1*p + k1][col] = 1
                W[k0*n + 0][col] = 1
                W[k1*n + 1][col] = 1
                col += 1
                # M2: (A10+A11)*B00
                U[2][col] = 1
                U[3][col] = 1
                V[0*p + k0][col] = 1
                W[k0*n + 1][col] = 1
                W[k1*n + 1][col] = -1
                col += 1
                # M3: A00*(B10-B00)
                U[0][col] = 1
                V[0*p + k1][col] = 1
                V[1*p + k1][col] = -1
                W[k1*n + 0][col] = 1
                W[k1*n + 1][col] = 1
                col += 1
                # M4: A11*(B01-B11)
                U[3][col] = 1
                V[1*p + k0][col] = 1
                V[0*p + k0][col] = -1
                W[k0*n + 0][col] = 1
                W[k0*n + 1][col] = 1
                col += 1
                # M5: (A00+A01)*B11
                U[0][col] = 1
                U[1][col] = 1
                V[1*p + k1][col] = 1
                W[k0*n + 0][col] = -1
                W[k1*n + 0][col] = 1
                col += 1
                # M6: (A10-A00)*(B00+B01)
                U[2][col] = 1
                U[0][col] = -1
                V[0*p + k0][col] = 1
                V[0*p + k1][col] = 1
                W[k1*n + 1][col] = 1
                col += 1
                # M7: (A01-A11)*(B10+B11)
                U[1][col] = 1
                U[3][col] = -1
                V[1*p + k0][col] = 1
                V[1*p + k1][col] = 1
                W[k0*n + 0][col] = 1
                col += 1
            
            # Add the 2x2x3 block for columns {6,7,8}
            # The 2x2x3 tensor uses columns 6,7,8 of the full 2x2x9 tensor.
            # In the 2x2x3 sub-tensor, column k corresponds to full column k+6.
            for r_idx in range(7):
                for i_idx in range(4):
                    U[i_idx][col] = U3[i_idx][r_idx]
                for j_idx in range(6):
                    # j_idx = j*3 + k_sub, where k_sub in {0,1,2}
                    # Full column = 6 + k_sub
                    # V in full tensor has entries V[j*9 + k_full][col]
                    # In the 2x2x3 sub-tensor, V[j_idx][r_idx] where j_idx = j*3+k_sub
                    # So j = j_idx // 3, k_sub = j_idx % 3
                    # Full: V[j*9 + (6+k_sub)][col]
                    j = j_idx // 3
                    k_sub = j_idx % 3
                    V[j * p + 6 + k_sub][col] = V3[j_idx][r_idx]
                for k_idx in range(6):
                    # k_idx = k_sub*2 + i_sub, where k_sub in {0,1,2}, i_sub in {0,1}
                    # In the 2x2x3 sub-tensor, W[k_idx][r_idx] where k_idx = k_sub*2+i_sub
                    # Full: W[(6+k_sub)*2 + i_sub][col]
                    k_sub = k_idx // 2
                    i_sub = k_idx % 2
                    W[(6 + k_sub) * n + i_sub][col] = W3[k_idx][r_idx]
                col += 1
            
            # Verify
            if _validate_fast(U, V, W, n, m, p):
                return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS

    # Try rank-28 first
    result_28 = _try_rank_28_cyclic(n, m, p)
    if result_28 is not None:
        U28, V28, W28 = result_28
        return {"U": U28, "V": V28, "W": W28}

    # Build rank-32 decomposition
    U, V, W = _build_rank_32(n, m, p)

    # Validate
    if _validate_fast(U, V, W, n, m, p):
        # Try to reduce rank
        reduced = _try_reduce_rank(U, V, W, n, m, p)
        if reduced is not None:
            U2, V2, W2 = reduced
            if _validate_fast(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}

        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
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
