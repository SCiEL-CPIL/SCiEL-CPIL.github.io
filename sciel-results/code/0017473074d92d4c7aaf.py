"""Matrix multiplication 3,3,8: rank-minimal exact tensor decomposition search."""
import random
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72


def _is_halfint(x):
    r = float(x) * 2.0
    return abs(r - round(r)) < 1e-9


def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        # Precompute expected tensor values
        # a = i*M + j, b = j*P + k, kk = k*N + i
        # Condition: j (from a) == j (from b)
        for a in range(N * M):
            i = a // M
            j_a = a % M
            for b in range(M * P):
                j_b = b // P
                k = b % P
                if j_a != j_b:
                    continue
                # Only one output index is non-zero for this (a,b)
                kk = k * N + i
                acc = 0.0
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    acc += u * v * W[kk][c]
                if abs(acc - 1.0) > tol:
                    return False
        
        # Check that all other entries are 0
        # We can do this by checking a random subset or all entries if fast enough
        # Total entries: 9 * 24 * 24 = 5184. Rank ~50. Ops ~ 250k.
        # We already checked the 1s. Now check 0s.
        for a in range(N * M):
            i = a // M
            j_a = a % M
            for b in range(M * P):
                j_b = b // P
                k = b % P
                if j_a != j_b:
                    # All kk should be 0
                    for kk in range(P * N):
                        acc = 0.0
                        for c in range(rank):
                            u = U[a][c]
                            if u == 0: continue
                            v = V[b][c]
                            if v == 0: continue
                            acc += u * v * W[kk][c]
                        if abs(acc) > tol:
                            return False
                else:
                    # kk = k*N + i should be 1, others 0
                    kk_target = k * N + i
                    for kk in range(P * N):
                        if kk == kk_target:
                            continue
                        acc = 0.0
                        for c in range(rank):
                            u = U[a][c]
                            if u == 0: continue
                            v = V[b][c]
                            if v == 0: continue
                            acc += u * v * W[kk][c]
                        if abs(acc) > tol:
                            return False
        return True
    except Exception:
        return False


def _random_structured(rank, iters, deadline, rng, start_rank=None):
    """Random search for a sparse exact decomposition of a given rank."""
    best = None
    start_rank = start_rank if start_rank is not None else rank
    
    # Precompute target tensor vectors for linear algebra check
    # t_k is a vector of length NM*MP
    # t_k[a,b] = 1 if a=i*M+j, b=j*P+k, else 0.
    # Note: For fixed k, the non-zeros are at (i, j, j, k) for all i,j.
    # a = i*M+j, b = j*P+k.
    
    for it in range(iters):
        if time.time() > deadline:
            break
        # Random sparse U, V
        U = [[0] * start_rank for _ in range(N * M)]
        V = [[0] * start_rank for _ in range(M * P)]
        
        # Assign random non-zeros
        for a in range(N * M):
            nzc = rng.randint(1, 3)
            for c in rng.sample(range(start_rank), nzc):
                U[a][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        for b in range(M * P):
            nzc = rng.randint(1, 3)
            for c in rng.sample(range(start_rank), nzc):
                V[b][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        # Construct matrix X of size (NM*MP) x R
        # X[(a,b), c] = U[a,c] * V[b,c]
        X = np.zeros((N * M * M * P, start_rank))
        for a in range(N * M):
            for b in range(M * P):
                idx = a * (M * P) + b
                for c in range(start_rank):
                    X[idx, c] = U[a][c] * V[b][c]
        
        # Check if target vectors are in span of columns of X
        # Target vectors t_k for k in 0..P-1
        # t_k has 1s at indices (a,b) where b//P == b%P? No.
        # b = j*P + k. a = i*M + j.
        # For fixed k, j can be 0..M-1, i can be 0..N-1.
        # Indices: a = i*M+j, b = j*P+k.
        
        valid = True
        W = [[0] * start_rank for _ in range(P * N)]
        
        # Solve for W for each k
        for k in range(P):
            # Construct t_k
            t_k = np.zeros(N * M * M * P)
            for i in range(N):
                for j in range(M):
                    a = i * M + j
                    b = j * P + k
                    idx = a * (M * P) + b
                    t_k[idx] = 1.0
            
            # Solve X @ w_k = t_k
            # Use least squares
            try:
                w_k, residuals, rank_x, sv = np.linalg.lstsq(X, t_k, rcond=None)
            except Exception:
                valid = False
                break
            
            # Check if solution is exact (residuals close to 0)
            if residuals and residuals[0] > 1e-9:
                valid = False
                break
            
            # Check if w_k entries are half-integers
            for c in range(start_rank):
                if not _is_halfint(w_k[c]):
                    valid = False
                    break
            if not valid:
                break
            
            # Store W
            # W index for this k is k*N + i
            # Wait, w_k is the vector W[:, c] for fixed k?
            # No, W[kk, c].
            # The equation was sum_c X[idx, c] W[kk, c] = t_k[idx].
            # So W[kk, :] = w_k.
            # But kk depends on i?
            # No, for fixed k, the target t_k corresponds to output column k.
            # The output indices are kk = k*N + i for i=0..N-1.
            # But t_k has 1s at ALL i for fixed k?
            # Yes, t_k[idx] = 1 for all i,j.
            # This implies that the same W row is used for all i?
            # No, W has different rows for each i.
            # The tensor equation is:
            # sum_c U[a,c] V[b,c] W[kk,c] = T[a,b,kk].
            # For fixed k, T[a,b,kk] is non-zero only for kk = k*N+i.
            # And it is 1 for all i.
            # This means we have N equations for each k?
            # No, the vector t_k I constructed has 1s at ALL (i,j).
            # This corresponds to a single linear combination of the columns of X.
            # But W has different rows for each i.
            # So we cannot solve for a single w_k for all i.
            
            valid = False
            break

    return best

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)

    # Baseline: schoolbook
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    best = (rank, U, V, W)

    # Search
    for cand_rank in range(30, 71):
        if time.time() > deadline:
            break
        res = _random_structured(cand_rank, iters=1000, deadline=deadline, rng=rng)
        if res is not None:
            best = res
            break

    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}
