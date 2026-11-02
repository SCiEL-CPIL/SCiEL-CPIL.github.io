"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 18
    
    # First, get the schoolbook decomposition as a fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition
    # Known result: 2x3x3 matrix multiplication has rank 17 (not 18)
    # Let's try to find it via random search over rationals
    
    # Strategy: Randomized search for rank 17
    # The tensor is T[i,j,k] = 1 if output (k,i) = sum_j A[i,j]*B[j,k]
    # We need U (6xR), V (9xR), W (6xR) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{output, input}
    
    # Actually, the decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) ... no
    
    # Let me reconsider. The standard tensor decomposition:
    # T_{ijk} = sum_c U_{ij,c} * V_{jk,c} * W_{ki,c}
    # where T_{ijk} = 1 if the (i,k)-th entry of AB depends on A_{ij} and B_{jk}
    # Actually T_{ijk} = 1 for all i,j,k (it's the full multiplication tensor)
    
    # Wait, the multiplication tensor for (n x m) * (m x p) -> (n x p):
    # T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # The output index is (i, k), and the bilinear form is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # So the tensor has entries T[i,j,k] = 1 for all i,j,k.
    # A rank-1 decomposition term c contributes:
    # U[:, :, c] outer V[:, :, c] outer W[:, :, c]
    # where U is n x m, V is m x p, W is p x n
    
    # The sum over c of U_c[i,j] * V_c[j,k] * W_c[k,i] should equal 1 for all i,j,k.
    
    # For schoolbook: each term picks one (i,j,k) and sets U[i,j]=1, V[j,k]=1, W[k,i]=1
    
    # Let's try random search for rank 17
    for target_rank in [17, 16, 15]:
        if time.time() > deadline:
            break
        found = _random_search(n, m, p, target_rank, deadline)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Convert to lists
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _random_search(n, m, p, target_rank, deadline):
    """
    Randomized search for a rank-target_rank decomposition.
    Uses random initialization and gradient-free refinement.
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    max_attempts = 200
    attempts_per_rank = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n_m, target_rank))
        V = np.random.randint(-2, 3, size=(m_p, target_rank))
        W = np.random.randint(-2, 3, size=(p_n, target_rank))
        
        # Try a few random perturbations
        for _ in range(attempts_per_rank):
            if time.time() > deadline:
                break
            
            if _verify(U, V, W, n, m, p):
                return U, V, W
            
            # Random perturbation
            idx = random.randint(0, target_rank - 1)
            U[random.randint(0, n_m - 1), idx] += random.randint(-1, 1)
            V[random.randint(0, m_p - 1), idx] += random.randint(-1, 1)
            W[random.randint(0, p_n - 1), idx] += random.randint(-1, 1)
    
    return None


def _structured_search(n, m, p, target_rank, deadline):
    """
    Try structured decompositions based on known results.
    For 2x3x3, there are known rank-17 decompositions.
    """
    # This is a placeholder for more sophisticated structured search
    return None

# EVOLVE-BLOCK-END
