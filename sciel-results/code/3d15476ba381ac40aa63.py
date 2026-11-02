"""Program to find a low-rank exact tensor decomposition for 2x5 by 5x5 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P  # 50
TIME_LIMIT = 130  # seconds, leaving 20s margin before 150s kill

def _build_tensor():
    """Build the matrix multiplication tensor T of shape (N*M, M*P, P*N)."""
    T = np.zeros((N * M, M * P, P * N), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1
    return T

def _verify(U, V, W):
    """Verify that U @ V^T @ W^T equals the tensor T."""
    T = _build_tensor()
    # T = sum_c U[:, c] outer V[:, c] outer W[:, c]
    # Equivalent: for each (i, j), sum_k T[i,j,k] = ...
    # Actually T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Let's verify by checking all entries
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We can check a few random entries or all
    for a in range(N * M):
        for b in range(M * P):
            # T[a,b,:] should be sum_r U[a,r] V[b,r] W[:,r]
            # Let's compute the vector
            vec = (U_arr[a, :] * V_arr[b, :]) @ W_arr.T  # shape (P*N,)
            if not np.array_equal(vec, T[a, b, :]):
                return False
    return True

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return (U.tolist(), V.tolist(), W.tolist())

def _try_random(rank, trials=200, deadline=None):
    """Try random integer/half-integer entries for given rank."""
    # Scale by 2 to use integers, then divide by 2 at the end
    # Entries are in {0, 1/2, 1, 3/2, 2} or similar
    max_val = 4  # max half-integer value
    for t in range(trials):
        if deadline and time.time() > deadline:
            break
        U = np.random.randint(0, max_val + 1, (N * M, rank))
        V = np.random.randint(0, max_val + 1, (M * P, rank))
        W = np.random.randint(0, max_val + 1, (P * N, rank))
        # Half-integers: multiply by 2 for verification?
        # Actually, let's just use integers for now and see if we can find a solution
        # The task says integer/half-integer entries
        # For verification, we can use integer arithmetic if we multiply all by 2
        # But let's just check with floats
        Uf = U.astype(float) / 2.0
        Vf = V.astype(float) / 2.0
        Wf = W.astype(float) / 2.0
        # Check a few entries for speed
        T = _build_tensor()
        valid = True
        for _ in range(50):
            a = random.randint(0, N * M - 1)
            b = random.randint(0, M * P - 1)
            vec = (Uf[a, :] * Vf[b, :]) @ Wf.T
            if not np.allclose(vec, T[a, b, :]):
                valid = False
                break
        if valid:
            # Full verification
            if _verify(Uf, Vf, Wf):
                return Uf, Vf, Wf
    return None

def _try_greedy(rank, deadline=None):
    """Try a greedy approach: start with rank-1 terms and add them."""
    # This is a simplified version of tensor decomposition
    T = _build_tensor()
    # We want to express T as sum of rank-1 tensors
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Initialize
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    # Greedy: find the largest rank-1 component of the residual
    residual = T.copy()
    for r in range(rank):
        if deadline and time.time() > deadline:
            break
        # Find a, b, c that maximize |residual[a,b,c]|
        # But we need to find a rank-1 tensor that best fits the residual
        # This is NP-hard in general, so let's use a heuristic
        # Try all possible (a, b) pairs and find the best c vector
        best_score = -1
        best_a, best_b, best_vec = None, None, None
        for a in range(N * M):
            for b in range(M * P):
                # residual[a,b,:] is the vector we want to match
                # We want to find a rank-1 tensor u * v * w
                # where u[a] = 1, v[b] = 1, and w = residual[a,b,:]
                # But this might not be the best choice
                # Let's just try to match residual[a,b,:] with w
                w_candidate = residual[a, b, :]
                # Check if this w_candidate is useful
                score = np.sum(w_candidate ** 2)
                if score > best_score:
                    best_score = score
                    best_a, best_b, best_vec = a, b, w_candidate
        if best_a is None or best_score < 1e-10:
            break
        # Add this rank-1 term
        U[best_a, r] = 1
        V[best_b, r] = 1
        W[:, r] = best_vec
        residual[best_a, best_b, :] -= best_vec
    return U, V, W

def construct():
    """Construct the factor matrices for matrix multiplication."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 10  # 10s margin
    
    # First, try the schoolbook decomposition as a baseline
    rank_schoolbook = N * M * P
    U_sb = np.zeros((N * M, rank_schoolbook))
    V_sb = np.zeros((M * P, rank_schoolbook))
    W_sb = np.zeros((P * N, rank_schoolbook))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_sb[i * M + j, col] = 1
                V_sb[j * P + k, col] = 1
                W_sb[k * N + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if _verify(U_sb, V_sb, W_sb):
        best_rank = rank_schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    else:
        best_rank = rank_schoolbook + 1
        best_U = best_V = best_W = None
    
    # Try to find a lower rank
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        # Try random search
        result = _try_random(rank, trials=100, deadline=deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    if best_U is None:
        # Fallback to schoolbook
        return _to_lists(U_sb, V_sb, W_sb)
    
    return _to_lists(best_U, best_V, best_W)

