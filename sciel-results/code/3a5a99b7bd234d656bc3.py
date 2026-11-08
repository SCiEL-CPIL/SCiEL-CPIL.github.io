<<<<<<< SEARCH
def _structured_init(n, m, p, target_rank, rng, mode=0):
    """Generate structured initializations exploiting the tensor structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if mode == 0:
        # Random initialization with scaling
        scale = 1.0 / (target_rank ** 0.25)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    elif mode == 1:
        # Structured: exploit that middle dimension is small (m=3)
        # For each j in 0..m-1, create a block of columns
        cols_per_j = target_rank // m
        for j in range(m):
            start = j * cols_per_j
            end = start + cols_per_j
            # U rows: i*m + j for i in 0..n-1
            for i in range(n):
                U[i * m + j, start:end] = rng.randn(cols_per_j) * 0.5
            # V rows: j*p + k for k in 0..p-1
            for k in range(p):
                V[j * p + k, start:end] = rng.randn(cols_per_j) * 0.5
            # W rows: k*n + i for k in 0..p-1, i in 0..n-1
            for k in range(p):
                for i in range(n):
                    W[k * n + i, start:end] = rng.randn(cols_per_j) * 0.5
        # Add some cross-terms
        remaining = target_rank - m * cols_per_j
        if remaining > 0:
            U[:, m * cols_per_j:] = rng.randn(n * m, remaining) * 0.3
            V[:, m * cols_per_j:] = rng.randn(m * p, remaining) * 0.3
            W[:, m * cols_per_j:] = rng.randn(p * n, remaining) * 0.3
    elif mode == 2:
        # Sparse-ish initialization with random support
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        # Zero out some entries to create structure
        for _ in range(target_rank // 3):
            col = rng.randint(target_rank)
            U[rng.choice(n * m, size=n * m // 3, replace=False), col] = 0
            V[rng.choice(m * p, size=m * p // 3, replace=False), col] = 0
            W[rng.choice(p * n, size=p * n // 3, replace=False), col] = 0
    elif mode == 3:
        # Initialize from perturbed schoolbook
        U0, V0, W0 = _schoolbook(n, m, p)
        idx = rng.choice(n * m * p, size=target_rank, replace=False)
        U = U0[:, idx] + rng.randn(n * m, target_rank) * 0.2
        V = V0[:, idx] + rng.randn(m * p, target_rank) * 0.2
        W = W0[:, idx] + rng.randn(p * n, target_rank) * 0.2
    elif mode == 4:
        # Kronecker-like structure: U = A ⊗ B pattern
        # For n=2, m=3, p=7: try to exploit the factorization
        A = rng.randn(n, m) * 0.5
        B = rng.randn(m, p) * 0.5
        # Create rank-m decomposition of A⊗B-like structure
        for j in range(m):
            start = j * (target_rank // m)
            end = start + (target_rank // m)
            U[j:end + j * (target_rank // m), :] = 0  # This is getting complex
            # Simpler: just use structured random
        U = rng.randn(n * m, target_rank) * 0.4
        V = rng.randn(m * p, target_rank) * 0.4
        W = rng.randn(p * n, target_rank) * 0.4
        # Ensure some rows have larger values
        for i in range(n):
            for j in range(m):
                U[i * m + j, :] *= 1.5
    else:
        # Default random
        scale = 1.0 / (target_rank ** 0.3)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    
    return U, V, W
=======
def _structured_init(n, m, p, target_rank, rng, mode=0):
    """Generate structured initializations exploiting the tensor structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if mode == 0:
        # Random initialization with scaling
        scale = 1.0 / (target_rank ** 0.25)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    elif mode == 1:
        # Structured: exploit that middle dimension is small (m=3)
        cols_per_j = max(1, target_rank // m)
        for j in range(m):
            start = j * cols_per_j
            end = min(start + cols_per_j, target_rank)
            for i in range(n):
                U[i * m + j, start:end] = rng.randn(end - start) * 0.5
            for k in range(p):
                V[j * p + k, start:end] = rng.randn(end - start) * 0.5
            for k in range(p):
                for i in range(n):
                    W[k * n + i, start:end] = rng.randn(end - start) * 0.5
        remaining = target_rank - m * cols_per_j
        if remaining > 0:
            U[:, m * cols_per_j:] = rng.randn(n * m, remaining) * 0.3
            V[:, m * cols_per_j:] = rng.randn(m * p, remaining) * 0.3
            W[:, m * cols_per_j:] = rng.randn(p * n, remaining) * 0.3
    elif mode == 2:
        # Sparse-ish initialization
        U = rng.randn(n * m, target_rank) * 0.3
        V = rng.randn(m * p, target_rank) * 0.3
        W = rng.randn(p * n, target_rank) * 0.3
        for _ in range(target_rank // 3):
            col = rng.randint(target_rank)
            U[rng.choice(n * m, size=max(1, n * m // 3), replace=False), col] = 0
            V[rng.choice(m * p, size=max(1, m * p // 3), replace=False), col] = 0
            W[rng.choice(p * n, size=max(1, p * n // 3), replace=False), col] = 0
    elif mode == 3:
        # Initialize from perturbed schoolbook
        U0, V0, W0 = _schoolbook(n, m, p)
        idx = rng.choice(n * m * p, size=target_rank, replace=False)
        U = U0[:, idx] + rng.randn(n * m, target_rank) * 0.2
        V = V0[:, idx] + rng.randn(m * p, target_rank) * 0.2
        W = W0[:, idx] + rng.randn(p * n, target_rank) * 0.2
    elif mode == 4:
        # Block-structured: assign columns to (i,j) pairs
        U = rng.randn(n * m, target_rank) * 0.4
        V = rng.randn(m * p, target_rank) * 0.4
        W = rng.randn(p * n, target_rank) * 0.4
        for i in range(n):
            for j in range(m):
                U[i * m + j, :] *= 1.5
    else:
        scale = 1.0 / (target_rank ** 0.3)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
    
    return U, V, W
>>>>>>> REPLACE

