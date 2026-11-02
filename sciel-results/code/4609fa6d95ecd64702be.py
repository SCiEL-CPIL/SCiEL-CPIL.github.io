"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_fast(rank, n, m, p, seed=42, max_iters=200):
    """Faster alternating minimization with vectorized operations."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    
    # Round to half-integers
    U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
    V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
    W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
    
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _try_rank(rank, n, m, p, seed=42, max_iters=500):
    """Try to find a decomposition of the given rank using random initialization."""
    np.random.seed(seed)
    # Initialize U, V, W with small random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] should satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # For each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                # Solve A_mat @ U[i*m+j, :] = b_vec
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    # Round to half-integers
    U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
    V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
    W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
    
    # Verify
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _try_rank_structured(rank, n, m, p, seed=42, max_iters=300):
    """Try structured initialization based on Kronecker patterns."""
    np.random.seed(seed)
    
    # Try different structured initializations
    for init_type in ['kronecker', 'block_diag', 'random_block']:
        if init_type == 'kronecker':
            # Initialize with Kronecker product structure
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Fill with random signs in blocks
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = np.random.choice([-1, 1]) * 0.5
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = np.random.choice([-1, 1]) * 0.5
                for i in range(n):
                    for k in range(p):
                        W[k*n+i, r] = np.random.choice([-1, 1]) * 0.5
        elif init_type == 'block_diag':
            # Block diagonal-like structure
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            block_size = max(1, rank // 4)
            for r in range(rank):
                block_start = (r // block_size) * block_size
                for i in range(n):
                    for j in range(m):
                        if (i*m+j) % rank < block_size:
                            U[i*m+j, r] = np.random.choice([-1, 1]) * 0.5
                for j in range(m):
                    for k in range(p):
                        if (j*p+k) % rank < block_size:
                            V[j*p+k, r] = np.random.choice([-1, 1]) * 0.5
                for i in range(n):
                    for k in range(p):
                        if (k*n+i) % rank < block_size:
                            W[k*n+i, r] = np.random.choice([-1, 1]) * 0.5
        else:  # random_block
            # Random sparse blocks
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        if np.random.random() < 0.3:
                            U[i*m+j, r] = np.random.choice([-1, 1]) * 0.5
                for j in range(m):
                    for k in range(p):
                        if np.random.random() < 0.3:
                            V[j*p+k, r] = np.random.choice([-1, 1]) * 0.5
                for i in range(n):
                    for k in range(p):
                        if np.random.random() < 0.3:
                            W[k*n+i, r] = np.random.choice([-1, 1]) * 0.5
        
        # Alternating minimization
        for it in range(max_iters):
            # Fix V, W, optimize U
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    b_vec = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Fix U, W, optimize V
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    b_vec = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Fix U, V, optimize W
            for i in range(n):
                for k in range(p):
                    A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    b_vec = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
        
        # Round to half-integers
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        # Verify
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    for rank in range(75, 55, -1):
        if time.time() > deadline - 30:
            break
        
        # Try random initialization first
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iters=150)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        # If not found, try structured initialization
        if best_rank == rank_schoolbook and time.time() < deadline - 15:
            for seed in range(5):
                if time.time() > deadline - 15:
                    break
                
                result = _try_rank_structured(rank, n, m, p, seed=seed, max_iters=100)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
# EVOLVE-BLOCK-END
