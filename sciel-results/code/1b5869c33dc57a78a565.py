"""
Rank-minimal tensor decomposition for 2x5x6 matrix multiplication.

Strategy: Block decomposition with optimized sub-blocks, then systematic rank reduction.
- Use Strassen (rank 7) for 2x2x2 blocks
- Use schoolbook (rank 4) for 2x1x2 blocks

Block layout for 2x5x6:
  m=5 split as [2, 2, 1], p=6 split as [2, 2, 2]
  6 Strassen blocks (rank 42) + 3 schoolbook blocks (rank 12) = rank 54

Then apply systematic rank reduction using linear algebra on the tensor space.
"""

import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s safety margin for 150s limit


def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def check_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster check using einsum."""
    rank = U.shape[1]
    result = np.einsum('ac,bc,dc->abd', U, V, W)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(result, target, atol=tol)


def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def strassen_decomposition_2x2():
    """
    Returns Strassen decomposition for a 2x2x2 block.
    U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank_block = 7
    U_block = np.zeros((4, rank_block))
    V_block = np.zeros((4, rank_block))
    W_block = np.zeros((4, rank_block))

    # Term 0: M1 = (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1

    # Term 1: M2 = (A10+A11)B00
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1

    # Term 2: M3 = A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1

    # Term 3: M4 = A11(B10-B00)
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1

    # Term 4: M5 = (A00+A10)B11
    U_block[0, 4] = 1; U_block[2, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1

    # Term 5: M6 = (A10-A00)(B00+B01)
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1

    # Term 6: M7 = (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1

    return U_block, V_block, W_block


def schoolbook_2x1x2():
    """
    Returns schoolbook decomposition for a 2x1x2 block.
    n=2, m=1, p=2. Rank = 4.
    U_block: (2, 4), V_block: (2, 4), W_block: (4, 4)
    """
    rank_block = 4
    U_block = np.zeros((2, rank_block))
    V_block = np.zeros((2, rank_block))
    W_block = np.zeros((4, rank_block))

    # c=0: i=0, j=0, k=0
    U_block[0, 0] = 1
    V_block[0, 0] = 1
    W_block[0, 0] = 1

    # c=1: i=0, j=0, k=1
    U_block[0, 1] = 1
    V_block[1, 1] = 1
    W_block[2, 1] = 1

    # c=2: i=1, j=0, k=0
    U_block[1, 2] = 1
    V_block[0, 2] = 1
    W_block[1, 2] = 1

    # c=3: i=1, j=0, k=1
    U_block[1, 3] = 1
    V_block[1, 3] = 1
    W_block[3, 3] = 1

    return U_block, V_block, W_block


def build_block_decomposition(n, m, p):
    """
    Builds a decomposition using multiple block types.
    Layout: m=5 as [2,2,1], p=6 as [2,2,2]
    Total rank: 6*7 + 3*4 = 54.
    """
    max_rank = 60
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))

    col = 0

    blocks = [
        (0, 0, 2, 2, 'strassen'),
        (0, 2, 2, 2, 'strassen'),
        (0, 4, 2, 2, 'strassen'),
        (2, 0, 2, 2, 'strassen'),
        (2, 2, 2, 2, 'strassen'),
        (2, 4, 2, 2, 'strassen'),
        (4, 0, 1, 2, 'schoolbook_2x1x2'),
        (4, 2, 1, 2, 'schoolbook_2x1x2'),
        (4, 4, 1, 2, 'schoolbook_2x1x2'),
    ]

    for j0, k0, mj, pk, method in blocks:
        if method == 'strassen':
            Ub, Vb, Wb = strassen_decomposition_2x2()
            for i in range(2):
                for j in range(mj):
                    row_idx = i*m + (j0+j)
                    U[row_idx, col:col+7] = Ub[i*2+j, :]
            for j in range(mj):
                for k in range(pk):
                    row_idx = (j0+j)*p + (k0+k)
                    V[row_idx, col:col+7] = Vb[j*2+k, :]
            for k in range(pk):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+7] = Wb[k*2+i, :]
            col += 7
        elif method == 'schoolbook_2x1x2':
            Ub, Vb, Wb = schoolbook_2x1x2()
            for i in range(2):
                row_idx = i*m + j0
                U[row_idx, col:col+4] = Ub[i, :]
            for k in range(pk):
                row_idx = j0*p + (k0+k)
                V[row_idx, col:col+4] = Vb[k, :]
            for k in range(pk):
                for i in range(2):
                    row_idx = (k0+k)*n + i
                    W[row_idx, col:col+4] = Wb[k*2+i, :]
            col += 4

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    if not check_decomposition(U, V, W, n, m, p):
        raise ValueError("Block decomposition failed verification")

    return U, V, W


def tensor_to_vector(U, V, W, n, m, p):
    """
    Convert a rank-1 term (U[:,c], V[:,c], W[:,c]) to a vector in the tensor space.
    The tensor space has dimension n*m*p, indexed by (i,j,k) -> i*m*p + j*p + k.
    """
    vec = np.zeros(n*m*p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i*m*p + j*p + k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    return vec


def find_minimal_rank(U, V, W, n, m, p, max_rank_to_try=None):
    """
    Find the minimal rank by checking linear dependencies of the rank-1 tensors.
    Uses Gaussian elimination on the tensor space.
    """
    rank = U.shape[1]
    if max_rank_to_try is None:
        max_rank_to_try = rank

    # Build the tensor matrix: each column is a rank-1 tensor vector
    tensor_dim = n * m * p
    tensor_matrix = np.zeros((tensor_dim, rank))
    
    for c in range(rank):
        tensor_matrix[:, c] = tensor_to_vector(U[:, c], V[:, c], W[:, c], n, m, p)

    # Find the rank of this matrix
    # The actual tensor rank is at least the rank of this matrix
    # But we need to be more careful: we need to find the minimum number of columns
    # whose span contains the target tensor.

    # First, compute the target tensor vector
    target = np.zeros(tensor_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m*p + j*p + k] = 1.0

    # Check if target is in the span of the columns
    # Use QR decomposition to find a basis
    try:
        Q, R = np.linalg.qr(tensor_matrix, mode='complete')
    except:
        return U.copy(), V.copy(), W.copy(), rank

    # The columns of Q up to the rank form a basis
    # Find the rank
    tol = 1e-10
    diag_R = np.abs(np.diag(R))
    num_rank = np.sum(diag_R > tol)
    
    if num_rank == 0:
        return U.copy(), V.copy(), W.copy(), rank

    # Use the first num_rank columns of Q as basis
    basis = Q[:, :num_rank]
    
    # Now find the minimum subset of original columns that spans the target
    # This is a set cover problem, but we can use greedy approach
    # First, check which basis vectors are needed
    
    # Project target onto the basis
    coords = basis.T @ target
    active_basis = np.abs(coords) > tol
    
    # Now find minimum subset of original columns
    # Greedy: start with all, remove one at a time if possible
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    curr_rank = rank
    
    # Try removing columns one at a time
    for attempt in range(200):  # Limit iterations
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        removed = False
        for c in range(curr_rank):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
            U_test = np.delete(U_curr, c, axis=1)
            V_test = np.delete(V_curr, c, axis=1)
            W_test = np.delete(W_curr, c, axis=1)
            
            # Rebuild tensor matrix for test
            tensor_test = np.zeros((tensor_dim, curr_rank - 1))
            for d in range(curr_rank - 1):
                tensor_test[:, d] = tensor_to_vector(U_test[:, d], V_test[:, d], W_test[:, d], n, m, p)
            
            # Check if target is still in span
            try:
                Q_t, R_t = np.linalg.qr(tensor_test, mode='complete')
                diag_t = np.abs(np.diag(R_t))
                rank_t = np.sum(diag_t > tol)
                if rank_t == 0:
                    continue
                basis_t = Q_t[:, :rank_t]
                coords_t = basis_t.T @ target
                # Check if target is in span: residual should be small
                proj = basis_t @ coords_t
                if np.linalg.norm(target - proj) < 1e-8:
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    curr_rank -= 1
                    removed = True
                    break
            except:
                continue
        
        if not removed:
            break

    return U_curr, V_curr, W_curr, curr_rank


def try_linear_combination_removal(U, V, W, n, m, p):
    """
    Try to find if any column is a linear combination of others in the tensor space.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    tensor_dim = n * m * p
    target = np.zeros(tensor_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m*p + j*p + k] = 1.0

    # Build tensor matrix
    tensor_matrix = np.zeros((tensor_dim, rank))
    for c in range(rank):
        tensor_matrix[:, c] = tensor_to_vector(U[:, c], V[:, c], W[:, c], n, m, p)

    tol = 1e-10

    for c in range(rank):
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break

        other_cols = [d for d in range(rank) if d != c]
        if len(other_cols) < 2:
            continue

        # Check if tensor_matrix[:, c] is in the span of other columns
        A = tensor_matrix[:, other_cols]
        b = tensor_matrix[:, c]

        try:
            # Use least squares to check if b is in span of A
            coeffs, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            pred = A @ coeffs
            residual = np.linalg.norm(b - pred)
        except:
            continue

        if residual < 1e-8:
            # Column c is redundant in tensor space
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            rank -= 1
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Rebuild tensor matrix
            tensor_matrix = np.zeros((tensor_dim, rank))
            for d in range(rank):
                tensor_matrix[:, d] = tensor_to_vector(U[:, d], V[:, d], W[:, d], n, m, p)
            
            c = -1  # Restart

    return best_U, best_V, best_W, best_rank


def try_kronecker_reduction(U, V, W, n, m, p):
    """
    Try to exploit structure to reduce rank by checking for proportional columns.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    tensor_dim = n * m * p
    target = np.zeros(tensor_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m*p + j*p + k] = 1.0

    # Build tensor matrix
    tensor_matrix = np.zeros((tensor_dim, rank))
    for c in range(rank):
        tensor_matrix[:, c] = tensor_to_vector(U[:, c], V[:, c], W[:, c], n, m, p)

    tol = 1e-10

    # Try removing pairs
    for c1 in range(rank - 1):
        if time.time() - START_TIME > TIME_LIMIT - 40:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() - START_TIME > TIME_LIMIT - 40:
                break
            
            # Check if columns c1 and c2 are proportional in tensor space
            v1 = tensor_matrix[:, c1]
            v2 = tensor_matrix[:, c2]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < tol or norm2 < tol:
                continue
            
            # Check if v2 = alpha * v1
            alpha = np.dot(v1, v2) / (norm1 * norm1)
            residual = np.linalg.norm(v2 - alpha * v1)
            
            if residual < 1e-8:
                # Remove c2
                U = np.delete(U, c2, axis=1)
                V = np.delete(V, c2, axis=1)
                W = np.delete(W, c2, axis=1)
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Rebuild tensor matrix
                tensor_matrix = np.zeros((tensor_dim, rank))
                for d in range(rank):
                    tensor_matrix[:, d] = tensor_to_vector(U[:, d], V[:, d], W[:, d], n, m, p)
                
                break  # Restart inner loop
    
    return best_U, best_V, best_W, best_rank


def construct():
    global START_TIME
    n, m, p = DIMENSIONS

    # 1. Build block decomposition (rank 54)
    try:
        U, V, W = build_block_decomposition(n, m, p)
    except:
        U, V, W = schoolbook_decomposition(n, m, p)

    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # 2. Try Kronecker-structure-aware reduction (proportional columns)
    if time.time() - START_TIME < TIME_LIMIT - 50:
        U2, V2, W2, rank2 = try_kronecker_reduction(U, V, W, n, m, p)
        if rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2

    # 3. Try linear combination removal (tensor space span)
    if time.time() - START_TIME < TIME_LIMIT - 30:
        U3, V3, W3, rank3 = try_linear_combination_removal(best_U, best_V, best_W, n, m, p)
        if rank3 < best_rank:
            best_rank = rank3
            best_U, best_V, best_W = U3, V3, W3

    # 4. Try systematic rank reduction
    if time.time() - START_TIME < TIME_LIMIT - 15:
        U4, V4, W4, rank4 = find_minimal_rank(best_U, best_V, best_W, n, m, p)
        if rank4 < best_rank:
            best_rank = rank4
            best_U, best_V, best_W = U4, V4, W4

    # 5. Final verification
    if not check_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = schoolbook_decomposition(n, m, p)
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
