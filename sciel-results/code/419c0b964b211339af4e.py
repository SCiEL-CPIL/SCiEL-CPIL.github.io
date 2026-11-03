"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = 150.0  # wall-clock limit in seconds
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _validate(U, V, W, n, m, p):
    """Check if U,V,W of given shapes exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = delta[i+k, j] in appropriate form
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta condition
    # Actually, the tensor for matrix mult: T_{i,j,k} where result[i,k] = sum_j A[i,j]*B[j,k]
    # Standard form: sum_c U[i,j,c] V[j,k,c] W[i,k,c] = delta
    # With indices: U indexed by (i,j), V by (j,k), W by (k,i) per trace convention W[k*n+i, r]
    
    # Reconstruct: for each (i,j,k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k form a valid "path" in the tensor, 0 otherwise
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the 
    # appropriate tensor product sense). Actually no - the tensor is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C ⊗ e_{i,k}^{AB}
    # In the factorization: sum_c U[i,j,c] V[j,k,c] W[k,i,c] should reproduce this.
    # For exact decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k
    # This is a 3D tensor contraction
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                result[i,j,k] = val
    
    # For matrix multiplication, the tensor should be all 1s
    # (each (i,j,k) triple contributes exactly one term)
    expected = np.ones((n, m, p))
    if not np.allclose(result, expected, atol=1e-9):
        return False
    return True

def _validate_fast(U, V, W, n, m, p):
    """Faster validation using numpy broadcasting."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient contraction
    # U[i*m+j, c] -> U_3d[i, j, c]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # result[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # Use einsum
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    expected = np.ones((n, m, p))
    return np.allclose(result, expected, atol=1e-9)

def _schoolbook(n, m, p):
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

def _try_ranked_search(n, m, p, max_rank, time_budget):
    """Try to find a decomposition with rank <= max_rank using structured search."""
    start = time.time()
    best = None
    
    # Strategy: Start from schoolbook and try to merge/eliminate terms
    # For small n,m,p, we can try random perturbations and check validity
    
    # For 6x6x6, known results suggest rank might be reducible
    # Try a few structured approaches within time budget
    
    while time.time() - start < time_budget:
        if _time_remaining() < 10:
            break
        
        # Approach: Try rank = max_rank with random sparse initialization
        rank = max_rank
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with random sparse patterns
        for c in range(rank):
            # Each term picks a random (i,j) for U, (j,k) for V, (k,i) for W
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        
        if _validate_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best

def construct():
    n, m, p = DIMENSIONS
    
    # Get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Check time budget
    if _time_remaining() < 30:
        return schoolbook
    
    # Try to find lower rank decompositions
    # Known: schoolbook rank is 216
    # Try a few structured lower ranks
    
    # For 6x6x6, try ranks from 216 down
    # Given time constraints, try a limited search
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try a quick structured search for rank reduction
    # Use the fact that matrix multiplication tensor has symmetries
    
    # Approach: Try to find a decomposition with rank < 216
    # by exploiting the structure
    
    # Time-boxed search
    search_time = min(100, _time_remaining() - 15)
    if search_time > 10:
        # Try random search for lower rank
        for target_rank in [215, 214, 213, 212, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120]:
            if _time_remaining() < 15:
                break
            if time.time() - _START_TIME > _DEADLINE - 15:
                break
            
            # Try a few random initializations
            for attempt in range(50):
                if _time_remaining() < 15:
                    break
                
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Initialize: start with schoolbook pattern and perturb
                # Each term corresponds to a (i,j,k) triple
                # For target_rank < 216, we need some terms to cover multiple triples
                
                # Simple random initialization
                for c in range(target_rank):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    U[i*m+j, c] = np.random.choice([-1, 1])
                    V[j*p+k, c] = np.random.choice([-1, 1])
                    W[k*n+i, c] = np.random.choice([-1, 1])
                
                if _validate_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
