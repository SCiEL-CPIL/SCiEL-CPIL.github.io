"""Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
import time
import random
import fractions
import numpy as np

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

DIMENSIONS = [3, 5, 5]


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _verify(U, V, W):
    """Check that the factor matrices exactly decompose the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,k}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if i == k else 0
                s = fractions.Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    uc = U[ui][c]
                    if uc == 0:
                        continue
                    vc = V[vj][c]
                    if vc == 0:
                        continue
                    wc = W[wk][c]
                    if wc == 0:
                        continue
                    s += fractions.Fraction(uc) * fractions.Fraction(vc) * fractions.Fraction(wc)
                if s != target:
                    return False
    return True


def _schoolbook():
    """Fallback: rank-75 schoolbook decomposition (always valid)."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _snap_and_verify(U, V, W):
    """Snap to half-integers and verify. Returns (valid, U, V, W) with snapped values."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify using exact arithmetic
    U_list = U_snap.tolist()
    V_list = V_snap.tolist()
    W_list = W_snap.tolist()
    
    if _verify(U_list, V_list, W_list):
        return True, U_list, V_list, W_list
    return False, None, None, None


def _try_low_rank_search(rank, seed, max_restarts=3):
    """
    Try to find a decomposition of the given rank using numerical optimization,
    then snap to half-integers and verify exactly.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.RandomState(seed)
    
    # Precompute target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T_target[I, J, K] = 1.0
    
    for restart in range(max_restarts):
        if _time_left() < 15.0:
            break
        
        # Random initialization
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5
        
        # Simple gradient descent with restarts
        lr = 0.01
        for step in range(200):
            if _time_left() < 10.0:
                break
            
            # Compute residual: Res[I,J,K] = sum_c U[I,c]*V[J,c]*W[K,c] - T_target
            Res = np.einsum('ic,jc,kc->ijk', U, V, W) - T_target
            cost = np.sum(Res**2)
            
            if cost < 1e-10:
                break
            
            # Compute gradients
            # dRes/dU[I,c] = sum_{J,K} Res[I,J,K] * V[J,c] * W[K,c]
            dU = np.einsum('ijk,jc,kc->ic', Res, V, W)
            dV = np.einsum('ijk,ic,kc->jc', Res, U, W)
            dW = np.einsum('ijk,ic,jc->kc', Res, U, V)
            
            # Gradient step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we converged
        Res = np.einsum('ic,jc,kc->ijk', U, V, W) - T_target
        if np.sum(Res**2) < 1e-8:
            valid, U_list, V_list, W_list = _snap_and_verify(U, V, W)
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    # Start with the known valid schoolbook decomposition (rank 75)
    best = _schoolbook()
    best_rank = 75
    
    # Try lower ranks using numerical optimization + snapping
    # The theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Try a range of ranks between 25 and 74
    
    ranks_to_try = list(range(25, 75, 5))  # 25, 30, 35, ..., 70
    # Add some finer-grained tries near the lower bound
    ranks_to_try = [25, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    
    for rank in ranks_to_try:
        if _time_left() < 20.0:
            break
        
        # Try a few random seeds for this rank
        for seed in range(3):
            if _time_left() < 10.0:
                break
            
            result = _try_low_rank_search(rank, seed, max_restarts=2)
            if result is not None:
                best = result
                best_rank = rank
                break  # Found a valid decomposition at this rank, try lower
        
        if best_rank < 75:
            break  # Found an improvement, stop searching higher ranks
    
    return best


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
