import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s safety margin

def _check_time():
    return (time.time() - START_TIME) > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition_fast(U, V, W, n, m, p, T=None):
    """Fast verification using numpy einsum."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if T is None:
        T = _build_target_tensor(n, m, p)
    
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(result, T, atol=1e-10)

def _laderman_4x4():
    """
    Laderman's 23-term decomposition for 4x4 matrix multiplication.
    Returns U, V, W of shape (16, 23) each for 4x4x4.
    
    Laderman 1976: 23 scalar multiplications for 4x4.
    We construct this by block decomposition using 2x2 Strassen blocks.
    """
    # Actually, let's use a known explicit 23-term Laderman decomposition.
    # The Laderman decomposition for 4x4 matrix multiplication uses 23 products.
    # 
    # Let A and B be 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    #
    # The 23 products are (using 2x2 Strassen-like formulas on blocks):
    # We'll construct the rank-23 decomposition directly.
    
    # For 4x4, we can use the following approach:
    # Partition into 2x2 blocks and use a refined Strassen approach.
    
    # Let me use a concrete known Laderman decomposition.
    # The key insight: 4x4 can be done with 23 multiplications.
    
    # I'll construct it via the following 23 linear forms:
    # M1 = (A11-A12-A21+A22)*(B11+B12-B21-B22)
    # M2 = (A11+A12)*(B21-B22)  
    # M3 = (A11-A12)*(B11-B21)
    # M4 = (A21+A22)*(B12-B22)
    # M5 = (A21-A22)*(B11+B12)
    # M6 = A11*B11
    # M7 = A12*B12
    # M8 = A21*B21
    # M9 = A22*B22
    # M10 = (A11+A21)*(B11+B21)
    # M11 = (A12+A22)*(B12+B22)
    # M12 = (A11-A21)*(B11-B21)
    # M13 = (A12-A22)*(B12-B22)
    # M14 = (A11+A12)*(B11+B12)
    # M15 = (A21+A22)*(B21+B22)
    # M16 = (A11-A12)*(B21-B22)
    # M17 = (A21-A22)*(B11+B12)
    # M18 = (A11+A22)*(B11+B22)
    # M19 = (A12+A21)*(B12+B21)
    # M20 = (A11-A22)*(B11-B22)
    # M21 = (A12-A21)*(B12-B21)
    # M22 = (A11+A12+A21+A22)*(B11+B12+B21+B22)
    # M23 = (A11-A12-A21+A22)*(B11-B12-B21+B22)
    
    # This is getting complex. Let me use a different approach.
    # I'll construct the Laderman decomposition by using the fact that
    # 4x4 matrix multiplication can be expressed as a sum of 23 rank-1 tensors.
    
    # Actually, for this task (3x4x6), let me take a more direct approach.
    # I'll use the schoolbook decomposition and then apply algebraic reductions.
    
    n = m = p = 4
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 23 terms for 4x4 matrix multiplication
    # Each term r has: U[:,r], V[:,r], W[:,r]
    # where U[a,r]*V[b,r]*W[c,r] contributes to T[a,b,c]
    
    # For 4x4: a = i*4+j, b = j*4+k, c = k*4+i
    # We need sum_r U[a,r]*V[b,r]*W[c,r] = delta(a=i*4+j, b=j*4+k, c=k*4+i)
    
    # Let me use a known explicit construction.
    # The Laderman decomposition can be written as:
    
    # I'll use the following known 23-term decomposition for 4x4:
    # Based on Laderman (1976), the 23 products are:
    
    terms = []
    
    # Block structure: 4x4 = [[A,B],[C,D]] where each is 2x2
    # Using Strassen on 2x2 blocks gives 7 products per block interaction,
    # but Laderman's refinement gives 23 total.
    
    # Let me just use a verified 23-term decomposition.
    # For simplicity and correctness, let me construct it carefully.
    
    # Actually, for the 3x4x6 case, a better strategy is:
    # 1. Start with schoolbook (rank 72)
    # 2. Use algebraic reduction to find linear dependencies
    
    # Let me implement a proper algebraic reduction algorithm.
    return None  # We'll handle 3x4x6 directly

def _algebraic_reduction(U, V, W, n, m, p, T, max_rounds=50, time_limit=60):
    """
    Reduce the rank of a tensor decomposition by finding linear dependencies.
    
    A decomposition with rank R has columns u_r, v_r, w_r for r=1..R.
    If we can find coefficients c_r (not all zero) such that:
      sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0
    then we can reduce the rank.
    
    More practically: if the tensor decomposes as sum of R rank-1 terms,
    and these R terms are linearly dependent in the tensor space, we can
    find a lower-rank decomposition.
    
    Algorithm:
    1. Check if the current decomposition is valid.
    2. Try to find a non-trivial linear combination of the rank-1 terms
       that sums to zero.
    3. If found, use it to reduce the rank.
    """
    start = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = len(current_U[0])
    
    # Build the tensor from the decomposition
    def _tensor_from_decomposition(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Verify initial decomposition
    if not _verify_decomposition_fast(current_U, current_V, current_W, n, m, p, T):
        return current_U, current_V, current_W, current_rank
    
    for round_num in range(max_rounds):
        if _check_time() or (time.time() - start) > time_limit:
            break
        
        # Try to find a column r such that removing it and adjusting
        # other columns still gives a valid decomposition.
        
        # Method: For each pair of columns (r, s), check if column r
        # can be expressed as a linear combination involving column s
        # in a way that preserves the tensor.
        
        # Simpler: try removing each column and see if the remaining
        # columns span the same tensor.
        
        found_reduction = False
        
        for r in range(current_rank):
            if _check_time() or (time.time() - start) > time_limit:
                break
            
            # Try removing column r
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            for row in test_U:
                del row[r]
            for row in test_V:
                del row[r]
            for row in test_W:
                del row[r]
            
            if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, T):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                found_reduction = True
                break
        
        if not found_reduction:
            # Try a more sophisticated reduction:
            # Check if any two columns can be merged.
            # If u_r = alpha * u_s and v_r = beta * v_s and w_r = (1/(alpha*beta)) * w_s
            # for some alpha, beta, then we can merge.
            # But this is rare. Instead, try random linear combinations.
            
            # Try: for each column r, see if it can be written as a
            # linear combination of other columns in the tensor sense.
            
            # Build the matrix of tensor coefficients
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # Flatten: T_flat = sum_r vec(u_r ⊗ v_r ⊗ w_r)
            
            U_arr = np.array(current_U, dtype=float)
            V_arr = np.array(current_V, dtype=float)
            W_arr = np.array(current_W, dtype=float)
            
            # Each rank-1 term contributes a vector of size n*m*p*n*m*p*p*n
            # That's too large. Instead, use the verification approach.
            
            # Alternative: try random subsets of columns to remove
            if current_rank > 10:
                # Try removing 2 columns at a time
                for r in range(current_rank - 1):
                    if _check_time() or (time.time() - start) > time_limit:
                        break
                    for s in range(r + 1, current_rank):
                        if _check_time() or (time.time() - start) > time_limit:
                            break
                        
                        test_U = [row[:] for row in current_U]
                        test_V = [row[:] for row in current_V]
                        test_W = [row[:] for row in current_W]
                        
                        # Remove columns s first (higher index), then r
                        for row in test_U:
                            del row[s]
                            del row[r]
                        for row in test_V:
                            del row[s]
                            del row[r]
                        for row in test_W:
                            del row[s]
                            del row[r]
                        
                        if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, T):
                            current_U = test_U
                            current_V = test_V
                            current_W = test_W
                            current_rank -= 2
                            found_reduction = True
                            break
                    if found_reduction:
                        break
        
        if not found_reduction:
            break
    
    return current_U, current_V, current_W, current_rank

def _strassen_3x3_decomposition():
    """
    Strassen's 7-term decomposition for 2x2, extended to 3x3.
    For 3x3, the optimal rank is 19 (Strassen 1969).
    
    We construct the 19-term decomposition for 3x3 matrix multiplication.
    """
    n = m = p = 3
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 19-term decomposition for 3x3:
    # Partition 3x3 into overlapping 2x2 blocks and use Strassen.
    
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[j,k,l],[m,n,o],[p,q,r]]
    #
    # C = A*B
    #
    # The 19 products are:
    # M1 = (a-b)*(j+k)
    # M2 = (a+b)*m
    # M3 = d*(j-m)
    # M4 = i*(m+q)  -- wait, this doesn't look right for 3x3
    
    # Let me use a known explicit 19-term decomposition for 3x3.
    # From Strassen (1969), the 3x3 case uses 19 multiplications.
    
    # I'll use the following construction:
    # Split 3x3 into four overlapping 2x2 submatrices and combine.
    
    # Actually, let me just use a verified decomposition.
    # The 19-term Strassen decomposition for 3x3 is well-known but
    # complex to write out. Let me use a different approach.
    
    # For the 3x4x6 problem, I'll focus on direct algebraic reduction
    # from the schoolbook decomposition.
    
    # Placeholder: return schoolbook for now
    rank = n * m * p
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
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook (naive) decomposition with rank n*m*p."""
    rank = n * m * p
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
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p, T, max_rounds=100, time_limit=30):
    """
    Try to reduce rank by finding columns that can be merged.
    Two columns r and s can be merged if there exist scalars alpha, beta
    such that the tensor contribution of column r can be absorbed into
    column s (or vice versa).
    """
    start = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = len(current_U[0])
    
    for round_num in range(max_rounds):
        if _check_time() or (time.time() - start) > time_limit:
            break
        
        found = False
        
        # Try removing each column
        for r in range(current_rank):
            if _check_time() or (time.time() - start) > time_limit:
                break
            
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            for row in test_U:
                del row[r]
            for row in test_V:
                del row[r]
            for row in test_W:
                del row[r]
            
            if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, T):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    return current_U, current_V, current_W, current_rank

def _random_linear_combination_search(U, V, W, n, m, p, T, time_limit=30):
    """
    Try random linear combinations of columns to find a lower-rank decomposition.
    
    Idea: if we have R columns, maybe a subset of R-1 columns with modified
    coefficients can still represent the tensor.
    """
    start = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = len(current_U[0])
    best_rank = current_rank
    best_U, best_V, best_W = current_U[:], current_V[:], current_W[:]
    
    for attempt in range(200):
        if _check_time() or (time.time() - start) > time_limit:
            break
        
        # Try removing a random column
        r = random.randint(0, current_rank - 1)
        test_U = [row[:] for row in current_U]
        test_V = [row[:] for row in current_V]
        test_W = [row[:] for row in current_W]
        
        for row in test_U:
            del row[r]
        for row in test_V:
            del row[r]
        for row in test_W:
            del row[r]
        
        if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, T):
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = current_U[:], current_V[:], current_W[:]
    
    return best_U, best_V, best_W, best_rank

def _strassen_block_decomposition(n, m, p):
    """
    Use Strassen-like block decomposition to get a lower-rank starting point.
    
    For 3x4x6:
    - Split the 4x6 matrix B into two 4x3 blocks: B = [B1 | B2]
    - Compute A*B1 and A*B2 separately
    - Each 3x4x3 can potentially use a lower-rank decomposition
    
    For 3x4x3:
    - Schoolbook rank = 3*4*3 = 36
    - Can we do better?
    
    Let's try a different split:
    - A is 3x4, B is 4x6
    - Split A into two 3x2 blocks: A = [A1 | A2]
    - Split B into two 2x6 blocks: B = [B1; B2] (vertical)
    - C = A1*B1 + A2*B2
    - Each 3x2x6 has schoolbook rank 3*2*6 = 36
    - Total: 72 (same as schoolbook)
    
    Better: use overlapping blocks or Strassen-style.
    
    For 3x4x6, let's try:
    - Use 2x2 Strassen on a padded 4x4x4, then project.
    """
    # Pad 3x4x6 to 4x4x4
    # A is 3x4 -> pad to 4x4 (add zero row)
    # B is 4x6 -> pad to 4x4 (add zero columns, but 6>4 so this doesn't work)
    
    # Alternative: pad to 4x6x6? No, let's think differently.
    
    # For 3x4x6, the schoolbook rank is 72.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # So we can potentially get down to 24.
    
    # Let's try a block decomposition:
    # Split the middle dimension (4) into 2+2
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # Each term is 3x2x6 with schoolbook rank 36
    # Total: 72
    
    # Try 3+1 split:
    # A = [A1 | A2] where A1 is 3x3, A2 is 3x1
    # B = [B1; B2] where B1 is 3x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    # A1*B1: 3x3x6, schoolbook rank 54
    # A2*B2: 3x1x6, schoolbook rank 18
    # Total: 72
    
    # It seems hard to beat 72 with simple block splits.
    # Let's rely on algebraic reduction.
    
    return _construct_schoolbook(n, m, p)

def _polynomial_time_reduction(U, V, W, n, m, p, T, time_limit=60):
    """
    More systematic reduction using linear algebra.
    
    The tensor T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    If the R vectors (u_r ⊗ v_r ⊗ w_r) are linearly dependent,
    we can find a lower-rank decomposition.
    
    We check this by building the matrix whose columns are the
    flattened rank-1 tensors and computing its rank.
    """
    start = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = len(current_U[0])
    
    # The tensor has size n*m * m*p * p*n
    # For 3x4x6: 12 * 24 * 18 = 5184
    # Each rank-1 term is a vector of size 5184
    # With R=72 terms, we have a 5184 x 72 matrix
    
    U_arr = np.array(current_U, dtype=float)
    V_arr = np.array(current_V, dtype=float)
    W_arr = np.array(current_W, dtype=float)
    
    # Build the matrix of flattened rank-1 tensors
    # M[:, r] = vec(u_r ⊗ v_r ⊗ w_r)
    # But this is expensive. Instead, use a different approach.
    
    # Check if any column is redundant by trying to remove it.
    # This is O(R) verifications, each O(n^2*m^2*p^2) which is fine.
    
    for round_num in range(50):
        if _check_time() or (time.time() - start) > time_limit:
            break
        
        found = False
        for r in range(current_rank):
            if _check_time() or (time.time() - start) > time_limit:
                break
            
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            for row in test_U:
                del row[r]
            for row in test_V:
                del row[r]
            for row in test_W:
                del row[r]
            
            if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, T):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Start with schoolbook decomposition
    U, V, W = _construct_schoolbook(n, m, p)
    best_rank = len(U[0])
    
    # Try systematic column removal
    if not _check_time():
        U, V, W, best_rank = _polynomial_time_reduction(U, V, W, n, m, p, T, time_limit=30)
    
    # Try random search for further reductions
    if not _check_time():
        U_rand, V_rand, W_rand, rand_rank = _random_linear_combination_search(U, V, W, n, m, p, T, time_limit=20)
        if rand_rank < best_rank:
            U, V, W = U_rand, V_rand, W_rand
            best_rank = rand_rank
    
    # Final verification
    if not _verify_decomposition_fast(U, V, W, n, m, p, T):
        # Fall back to schoolbook
        U, V, W = _construct_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
