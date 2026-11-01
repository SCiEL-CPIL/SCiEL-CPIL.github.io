"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen on 2x2 blocks with linear dependency elimination.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with 0-based local indices."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = A00(B01-B11)
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = A11(B10-B00)
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    return U, V, W

def _matrices_to_terms(U, V, W, n, m, p):
    """Convert U, V, W matrices back to list of term dicts."""
    rank = U.shape[1]
    terms = []
    for c in range(rank):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for idx in range(n * m):
            if U[idx, c] != 0:
                i, j = divmod(idx, m)
                u_dict[(i, j)] = U[idx, c]
        for idx in range(m * p):
            if V[idx, c] != 0:
                j, k = divmod(idx, p)
                v_dict[(j, k)] = V[idx, c]
        for idx in range(p * n):
            if W[idx, c] != 0:
                k, i = divmod(idx, n)
                w_dict[(k, i)] = W[idx, c]
        terms.append((u_dict, v_dict, w_dict))
    return terms

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal delta_{i,k} for matrix mult (C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually for matrix multiplication, T[i,j,k] = 1 if this is a valid term
    # The tensor is: T[i,j,k] = 1 for all (i,j,k) where the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right form)
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Let me verify: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k.
    
    # Compute the decomposition
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # Compute sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This should equal 1 for all (i,j,k)
    # Use einsum: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p))
    
    max_err = np.max(np.abs(result - expected))
    return max_err < tol, max_err

def _reduce_rank(U, V, W, n, m, p, deadline=None):
    """Try to reduce the rank by finding linear dependencies among the terms.
    
    The idea: if the columns of U, V, W have linear dependencies, we can reduce rank.
    We look for a rank-revealing decomposition.
    
    Approach: 
    1. Check if the current decomposition is valid
    2. Try to find a lower-rank decomposition by looking at the tensor's structure
    3. Use iterative elimination: for each term, check if it can be expressed as a linear combination of others
    
    A more practical approach: use the fact that if we have more terms than necessary,
    some columns must be linearly dependent in a tensor sense.
    """
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 100.0
    
    rank = U.shape[1]
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: try to eliminate terms one at a time by checking if the remaining
    # terms can still reconstruct the tensor.
    # This is expensive, so we'll use a heuristic.
    
    # Better approach: look at the rank of the "flattened" tensors.
    # For each mode, the rank of the decomposition gives a lower bound.
    # But we want to find actual dependencies.
    
    # Practical approach: try random linear combinations to see if we can reduce.
    # Or: use the fact that for small problems, we can do exhaustive search.
    
    # Let's try a different approach: use SVD-like decomposition on the tensor.
    # Actually, let's try greedy elimination with verification.
    
    # First, let's see if there are any obviously dependent columns.
    # Check rank of U, V, W individually
    rank_U = np.linalg.matrix_rank(current_U, tol=1e-10)
    rank_V = np.linalg.matrix_rank(current_V, tol=1e-10)
    rank_W = np.linalg.matrix_rank(current_W, tol=1e-10)
    
    # The tensor rank must be at least max(rank_U, rank_V, rank_W)
    # But it could be higher.
    
    # Try to reduce by finding if any column can be removed.
    # This is O(rank) verifications, each O(n*m*p*rank)
    # For rank=46, n*m*p=50, that's 46*50*46 = ~106k operations per check
    # Total: 46 * 106k = ~5M operations, should be fast.
    
    reduced = True
    while reduced and time.time() < deadline - 10:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Try removing column c
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, err = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W

def _build_decomposition_strategy_1(n, m, p):
    """Strategy 1: Partition 5 = 2+2+1, use Strassen on 2x2x2 blocks."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen
    add_strassen(0, 0, 0)
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen
    add_strassen(0, 0, 2)
    # Block (i in {0,1}, j in {0,1}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen
    add_strassen(0, 2, 0)
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen
    add_strassen(0, 2, 2)
    # Block (i in {0,1}, j in {2,3}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (i in {0,1}, j in {4}, k in {0,1,2,3,4}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 5)
    
    return terms

def _build_decomposition_strategy_2(n, m, p):
    """Strategy 2: Partition 5 = 2+3, use Strassen where possible."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen
    add_strassen(0, 0, 0)
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen
    add_strassen(0, 0, 2)
    # Block (i in {0,1}, j in {0,1}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen
    add_strassen(0, 2, 0)
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen
    add_strassen(0, 2, 2)
    # Block (i in {0,1}, j in {2,3}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (i in {0,1}, j in {4}, k in {0,1}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 2)
    # Block (i in {0,1}, j in {4}, k in {2,3}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 2, 4)
    # Block (i in {0,1}, j in {4}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 4, 5)
    
    return terms

def _build_decomposition_strategy_3(n, m, p):
    """Strategy 3: Try all 2x2x2 Strassen blocks where possible."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Use Strassen for all 2x2x2 blocks, schoolbook for the rest
    # i in {0,1}, j in {0,1}, k in {0,1}: Strassen
    add_strassen(0, 0, 0)
    # i in {0,1}, j in {0,1}, k in {2,3}: Strassen
    add_strassen(0, 0, 2)
    # i in {0,1}, j in {0,1}, k in {4}: Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # i in {0,1}, j in {2,3}, k in {0,1}: Strassen
    add_strassen(0, 2, 0)
    # i in {0,1}, j in {2,3}, k in {2,3}: Strassen
    add_strassen(0, 2, 2)
    # i in {0,1}, j in {2,3}, k in {4}: Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # i in {0,1}, j in {4}, k in {0,1,2,3,4}: Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 5)
    
    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    # Try multiple strategies and pick the best
    strategies = [
        _build_decomposition_strategy_1,
        _build_decomposition_strategy_2,
        _build_decomposition_strategy_3,
    ]
    
    best_terms = None
    best_rank = float('inf')
    
    for strategy_fn in strategies:
        if time.time() > deadline - 30:
            break
        
        terms = strategy_fn(n, m, p)
        rank = len(terms)
        
        if rank < best_rank:
            best_rank = rank
            best_terms = terms
    
    if best_terms is None:
        # Fallback: pure schoolbook
        best_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_terms.append(
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    )
        best_rank = len(best_terms)
    
    # Convert to matrices
    U, V, W = _terms_to_matrices(best_terms, n, m, p)
    
    # Verify
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline=deadline)
        
        # Verify again after reduction
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Revert to pre-reduction
            U, V, W = _terms_to_matrices(best_terms, n, m, p)
    
    # Convert to list format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
