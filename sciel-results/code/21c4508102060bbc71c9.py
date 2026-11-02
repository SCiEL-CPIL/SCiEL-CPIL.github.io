import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s

    n, m, p = 2, 2, 8

    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()

    # Try to find a lower-rank decomposition via random search
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = _search_low_rank(target_rank, remaining - 5)
        if result is not None:
            return result

    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1

    # Convert to lists of integers to satisfy "integer/half-integer" strictly
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

def _search_low_rank(target_rank, time_budget):
    """Search for a lower-rank decomposition using random initialization and gradient descent."""
    n, m, p = 2, 2, 8
    rank = target_rank

    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5

    # Compute the target tensor
    T = _compute_tensor(n, m, p)

    start_time = time.time()
    max_time = time_budget

    # Gradient descent
    lr = 0.01
    prev_loss = np.inf
    patience = 100
    no_improve = 0

    while time.time() - start_time < max_time:
        # Compute current tensor
        T_current = _tensor_from_factors(U, V, W, n, m, p)

        # Compute loss
        loss = np.sum((T - T_current) ** 2)

        if loss < prev_loss - 1e-10:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            break

        prev_loss = loss

        # Compute gradients
        grad_U, grad_V, grad_W = _compute_gradients(U, V, W, T, n, m, p)

        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W

        # Check convergence
        if loss < 1e-8:
            # Snap to nearest integers
            U = np.round(U).astype(int)
            V = np.round(V).astype(int)
            W = np.round(W).astype(int)

            # Verify exact decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None

def _compute_tensor(n, m, p):
    """Compute the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] = 1
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _tensor_from_factors(U, V, W, n, m, p):
    """Compute tensor from factor matrices."""
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T += np.einsum('i,j,k->ijk', u, v, w)
    return T

def _compute_gradients(U, V, W, T_target, n, m, p):
    """Compute gradients for U, V, W."""
    rank = U.shape[1]
    T_current = _tensor_from_factors(U, V, W, n, m, p)
    diff = T_target - T_current

    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)

    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]

        # Gradient w.r.t. U[:, r]
        grad_U[:, r] = np.einsum('ijk,jk->i', diff, np.outer(v, w))
        # Gradient w.r.t. V[:, r]
        grad_V[:, r] = np.einsum('ijk,ik->j', diff, np.outer(u, w))
        # Gradient w.r.t. W[:, r]
        grad_W[:, r] = np.einsum('ijk,ij->k', diff, np.outer(u, v))

    return grad_U, grad_V, grad_W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    T = _compute_tensor(n, m, p)
    T_current = _tensor_from_factors(U, V, W, n, m, p)
    return np.allclose(T, T_current, atol=1e-10)
