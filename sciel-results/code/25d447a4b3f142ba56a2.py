"""Program for matrix multiplication 3,4,5 tensor rank minimization via search."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Precompute the target tensor as a sparse representation.
# T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# We want to find U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
# such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

# Since entries can be integer/half-integer, we can scale by 2 to work in integers.
# Let U', V', W' have integer entries (twice the actual values).
# Then sum_c U'[i*m+j, c] * V'[j*p+k, c] * W'[k*n+i, c] = 4 for all i,j,k?
# Wait, no. If U = U'/2, V = V'/2, W = W'/2, then:
# sum_c (U'/2)(V'/2)(W'/2) = (1/8) sum_c U' V' W' = 1
# So sum_c U'[i,j,c] V'[j,k,c] W'[k,i,c] = 8.

# Actually, let's think about this differently. The task says "integer/half-integer entries".
# This likely means each entry is in {0, 1/2, 1, -1/2, -1, ...} or similar.
# Let's work with scaled integers: multiply all entries by 2, so we need integer entries.
# Constraint: sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] = 8 for all i,j,k.

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    # Check shapes
    if len(U) != n * m: return False
    if len(V) != m * p: return False
    if len(W) != p * n: return False
    if any(len(row) != r for row in U): return False
    if any(len(row) != r for row in V): return False
    if any(len(row) != r for row in W): return False
    # Check the tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_scaled(U2, V2, W2):
    """Verify with scaled integer entries where product sum should be 8."""
    n, m, p = N, M, P
    r = len(U2[0])
    if len(U2) != n * m: return False
    if len(V2) != m * p: return False
    if len(W2) != p * n: return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                if s != 8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_result = None
    best_rank = RANK_SCHOOLBOOK
    
    # Initialize with schoolbook
    U0 = [[0] * RANK_SCHOOLBOOK for _ in range(N * M)]
    V0 = [[0] * RANK_SCHOOLBOOK for _ in range(M * P)]
    W0 = [[0] * RANK_SCHOOLBOOK for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U0[i * M + j][col] = 1
                V0[j * P + k][col] = 1
                W0[k * N + i][col] = 1
                col += 1
    best_result = {"U": U0, "V": V0, "W": W0}
    
    # Strategy: Use probabilistic random linear algebra to find lower-rank decompositions.
    # Fix U and V randomly, solve for W via Gaussian elimination over Q.
    # Check if W has entries that are multiples of 1/2.
    
    for r in range(59, 25, -1):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Random small integer entries for U and V
            U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(N * M)]
            V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(M * P)]
            
            # Solve for W: for each (i,j,k): sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = 1
            num_vars = P * N * r
            n_eq = N * M * P  # 60
            
            # Build augmented matrix
            aug = []
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        row = [Fraction(0)] * (num_vars + 1)
                        for c in range(r):
                            coeff = U[i*M+j][c] * V[j*P+k][c]
                            if coeff != 0:
                                var_idx = (k*N + i) * r + c
                                row[var_idx] = Fraction(coeff)
                        row[num_vars] = Fraction(1)
                        aug.append(row)
            
            # Gaussian elimination over Q
            rank_aug = 0
            pivot_cols = []
            for col_idx in range(num_vars):
                if rank_aug >= n_eq:
                    break
                # Find pivot
                pivot = -1
                for row_idx in range(rank_aug, n_eq):
                    if aug[row_idx][col_idx] != 0:
                        pivot = row_idx
                        break
                if pivot == -1:
                    continue
                # Swap
                aug[rank_aug], aug[pivot] = aug[pivot], aug[rank_aug]
                pivot_cols.append(col_idx)
                # Eliminate below
                pv = aug[rank_aug][col_idx]
                for row_idx in range(rank_aug + 1, n_eq):
                    if aug[row_idx][col_idx] != 0:
                        factor = aug[row_idx][col_idx] / pv
                        for cc in range(col_idx, num_vars + 1):
                            aug[row_idx][cc] -= factor * aug[rank_aug][cc]
                rank_aug += 1
            
            # Check consistency
            consistent = True
            for row_idx in range(rank_aug, n_eq):
                if aug[row_idx][num_vars] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Back substitution
            w_vec = [Fraction(0)] * num_vars
            for i in range(len(pivot_cols) - 1, -1, -1):
                col_idx = pivot_cols[i]
                s = aug[i][num_vars]
                for j in range(i + 1, len(pivot_cols)):
                    col2 = pivot_cols[j]
                    if w_vec[col2] != 0:
                        s -= aug[i][col2] * w_vec[col2]
                w_vec[col_idx] = s / aug[i][col_idx]
            
            # Check if all entries are multiples of 1/2
            valid = True
            for val in w_vec:
                if val.denominator not in (1, 2):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Exact verification
            ok = True
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        s = Fraction(0)
                        for c in range(r):
                            s += Fraction(U[i*M+j][c]) * Fraction(V[j*P+k][c]) * w_vec[(k*N+i)*r + c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                W = [[float(w_vec[(k*N+i)*r + c]) for c in range(r)] for k in range(P) for i in range(N)]
                return {"U": U, "V": V, "W": W}
    
    return best_result
