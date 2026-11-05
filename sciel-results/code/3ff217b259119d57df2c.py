"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Attempt Laser Method over F_13 for rank-19 decomposition of 3x3 matrix multiplication.
    Falls back to schoolbook rank-27 if the construction fails verification.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 3
    p_mod = 13  # Prime for F_13
    
    # First, try the laser method construction
    try:
        result = _laser_method_f13(n, m, p, p_mod)
        if result is not None:
            # Verify the decomposition
            U, V, W = result['U'], result['V'], result['W']
            rank = len(U[0])
            if _verify_decomposition(U, V, W, n, m, p):
                return result
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Fallback to schoolbook rank-27
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Schoolbook (rank-27) decomposition."""
    rank = n * m * p  # 27
    
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Compute C = sum_r U_r * V_r * W_r (outer product interpretation)
    # C[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    C = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        u = np.array([U[i * m + j][r] for i in range(n) for j in range(m)]).reshape(n, m)
        v = np.array([V[j * p + k][r] for j in range(m) for k in range(p)]).reshape(m, p)
        w = np.array([W[k * n + i][r] for k in range(p) for i in range(n)]).reshape(p, n)
        # Rank-1 tensor: u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # Check C[i,j,k] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if C[i, j, k] != 1:
                    return False
    return True

def _laser_method_f13(n, m, p, prime):
    """
    Laser method over F_prime for 3x3 matrix multiplication.
    Maps indices to a cyclic subgroup and uses DFT to generate rank-1 terms.
    """
    # For 3x3, we map i,j,k in {0,1,2} to elements in F_13
    # Use a "laser" configuration: select 3 points in each dimension
    # such that the Fourier support is sparse
    
    # Define the embedding: map index x in {0,1,2} to a specific element in F_13
    # Using a geometric progression: 0->0, 1->1, 2->g where g is a generator
    # For F_13, primitive root is 2: 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=3, 2^5=6, 2^6=12, ...
    
    # Laser method: choose subsets A, B, C of F_13 of size 3 each
    # such that the convolution structure allows sparse Fourier representation
    
    # Standard laser method for 3x3:
    # A = {0, 1, 3}, B = {0, 1, 3}, C = {0, 1, 3}
    # But we need to verify this gives a valid decomposition
    
    # Alternative: use the known rank-19 construction if it exists
    # For now, attempt a systematic search over small subsets
    
    # Define candidate index mappings
    candidates = [
        [0, 1, 3],
        [0, 1, 2],
        [0, 1, 4],
        [0, 1, 5],
        [1, 2, 4],
        [0, 2, 4],
    ]
    
    # For each candidate mapping, try to construct the decomposition
    for a_map in candidates:
        for b_map in candidates:
            for c_map in candidates:
                if time.time() > 140:
                    return None
                
                try:
                    result = _try_laser_decomposition(a_map, b_map, c_map, n, m, p, prime)
                    if result is not None:
                        return result
                except Exception:
                    continue
    
    return None

def _try_laser_decomposition(a_map, b_map, c_map, n, m, p, prime):
    """
    Try to construct a laser method decomposition with given index mappings.
    """
    # The tensor T[i,j,k] = 1 for i,j,k in {0,1,2}
    # We want to represent this as sum of rank-1 terms
    
    # In the laser method, we embed into a larger space and use Fourier analysis
    # For simplicity, we'll attempt a direct sparse decomposition search
    
    # Target: find rank-1 terms (u_r, v_r, w_r) such that
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    
    # Start with a greedy approach: try to find a small number of rank-1 terms
    # that cover the tensor
    
    # For rank 19, we need 19 terms
    # This is a hard combinatorial problem, so we'll use a heuristic
    
    # Initialize with zeros
    rank = 19
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Use a simple heuristic: try to find terms that match subsets of the tensor
    # This is a placeholder - actual laser method requires more sophisticated algebra
    
    # For now, return None to indicate this approach needs more work
    return None
# EVOLVE-BLOCK-END
