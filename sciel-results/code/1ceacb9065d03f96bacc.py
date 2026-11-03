import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
DEADLINE = 140

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    For matrix multiplication, T[i,j,k] = 1 if i,j,k are all valid indices, 0 otherwise.
    Actually, the matrix multiplication tensor is defined as:
    T_{i,j,k} = delta_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    So T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    """
    r = U.shape[1]
    # Compute the tensor via the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r_idx in range(r):
                    val += U[i*m + j, r_idx] * V[j*p + k, r_idx] * W[k*n + i, r_idx]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Faster verification using numpy.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    r = U.shape[1]
    # Reshape to 3D tensors
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    # Compute tensor contraction
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can do this via einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if all entries are 1.0
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

def _schoolbook_nm_p(n, m, p):
    """Create the schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, start_rank, max_rank, time_limit):
    """
    Try to find a decomposition with rank between start_rank and max_rank.
    Uses a randomized search approach.
    """
    start_time = time.time()
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try several random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random sparse patterns
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Each column of U, V, W should have some non-zero entries
            # For a valid decomposition, each term r contributes to T[i,j,k]
            # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Strategy: for each rank term, choose a random (i,j,k) pattern
            # But we need the sum to be exactly 1 for all (i,j,k)
            
            # Alternative: use the structure that each term is a rank-1 tensor
            # Let's try to build terms that cover the tensor
            
            # For now, use a simple heuristic: each term picks one (i,j,k) and sets
            # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
            # But that's just schoolbook. We need to find better patterns.
            
            # Let's try a different approach: use the fact that we can have
            # entries with values other than 0 and 1.
            
            # Random initialization with small values
            U[:] = np.random.randn(n * m, rank) * 0.5
            V[:] = np.random.randn(m * p, rank) * 0.5
            W[:] = np.random.randn(p * n, rank) * 0.5
            
            # Verify
            if _verify_decomposition_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W, "rank": rank}
        
        # If we got here, no valid decomposition found for this rank with random init
        # Try a more structured approach for lower ranks
    
    return None

def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W matrices for 2x2x2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, r] = coefficient of A[i,j] in M_r
    # V[j*p+k, r] = coefficient of B[j,k] in M_r
    # W[k*n+i, r] = coefficient of M_r in C[i,k]
    
    # M1: A00 + A11, B00 + B11
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2: A10 + A11, B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3: A00, B01 - B11
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1; W[2, 2] = 1  # C01, C11
    
    # M4: A11, B10 - B00
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5: A00 + A01, B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00, C01
    
    # M6: A10 - A00, B00 + B01
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # M7: A01 - A11, B10 + B11
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return {"U": U, "V": V, "W": W}

def _try_structured_decomposition(n, m, p):
    """
    Try to build a structured decomposition using Strassen for 2x2 blocks
    and schoolbook for the rest.
    """
    # For 3x4x5, we can try to decompose into blocks
    # This is complex, so let's just return schoolbook for now
    # and let the search find better solutions
    
    # Actually, let's try a simple approach: use the fact that
    # 3x4x5 can be viewed as a combination of smaller multiplications
    
    # For now, return None to indicate no structured solution found
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    schoolbook = _schoolbook_nm_p(n, m, p)
    best_rank = n * m * p
    best_result = {"U": schoolbook["U"].tolist(), 
                   "V": schoolbook["V"].tolist(), 
                   "W": schoolbook["W"].tolist()}
    
    # Try to find a better decomposition
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But in practice, the rank is likely higher
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 20
    # But we know that for small matrices, the rank is often close to schoolbook
    
    # Try a few ranks below schoolbook
    min_rank_to_try = max(n * m, m * p, p * n)  # 20
    
    # Try from min_rank_to_try up to schoolbook - 1
    # But this could be expensive, so limit the search
    time_limit = 100  # seconds
    
    # Try structured decomposition first
    structured = _try_structured_decomposition(n, m, p)
    if structured is not None:
        if structured["rank"] < best_rank:
            best_rank = structured["rank"]
            best_result = {"U": structured["U"].tolist(),
                          "V": structured["V"].tolist(),
                          "W": structured["W"].tolist()}
    
    # Try rank search
    # Start from a reasonable rank and go up
    # For 3x4x5, let's try ranks from 20 to 59 (schoolbook is 60)
    search_result = _try_rank_search(n, m, p, min_rank_to_try, best_rank - 1, time_limit)
    if search_result is not None:
        if search_result["rank"] < best_rank:
            best_rank = search_result["rank"]
            best_result = {"U": search_result["U"].tolist(),
                          "V": search_result["V"].tolist(),
                          "W": search_result["W"].tolist()}
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_result
    
    return best_result
