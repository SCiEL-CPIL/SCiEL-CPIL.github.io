"""Tensor decomposition for 2x3 x 3x7 matrix multiplication with rank minimization attempts."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
DEADLINE = 145.0  # Safety margin from 150s limit
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def remaining():
    return DEADLINE - elapsed()


def schoolbook_decomposition():
    """Generate the trivial schoolbook decomposition of rank 42."""
    n, m, p = 2, 3, 7
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
    
    return {"U": U, "V": V, "W": W}


def verify_decomposition(U, V, W, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    
    The matrix multiplication tensor T[i,j,k] = sum_t U[i,t]*V[j,t]*W[k,t]
    should equal the standard matrix mult tensor:
    T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication, else 0.
    
    For 2x3 * 3x7:
    - i in [0,6) (2*3 = 6 input rows, indexed as i*m+j)
    - j in [0,21) (3*7 = 21, indexed as j*p+k)
    - k in [0,14) (7*2 = 14 output, indexed as k*n+i_out)
    
    The tensor entry T[i_input, j_input, k_output] should be 1 iff
    the output k_output corresponds to multiplying input i with input j.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Compute the tensor: T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # This is a 3-way contraction
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # We can compute this as: for each t, outer product of U[:,t], V[:,t], W[:,t]
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for t in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
    
    # Now check against the standard matrix mult tensor
    # For matrix C = A * B where A is 2x3, B is 3x7:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # 
    # Index mapping:
    # i_input = i*m + j (row of A, col of A)
    # j_input = j*p + k (row of B, col of B)  
    # k_output = k*n + i (col of C, row of C)
    #
    # T[i_input, j_input, k_output] = 1 iff:
    #   i_input = i_A * m + j_A, j_input = j_B * p + k_B, k_output = k_C * n + i_C
    #   and j_A == j_B and i_A == i_C
    #   i.e., the A element A[i_A][j_A] multiplies with B[j_B][k_B] to give C[i_C][k_C]
    #   where j_A = j_B (the shared dimension) and i_A = i_C (the row of C)
    
    errors = 0
    max_err = 0
    for i_A in range(n):
        for j in range(m):
            for k in range(p):
                i_input = i_A * m + j
                j_input = j * p + k
                k_output = k * n + i_A
                expected = 1.0
                actual = T[i_input, j_input, k_output]
                err = abs(actual - expected)
                if err > tol:
                    errors += 1
                    max_err = max(max_err, err)
    
    # Also check that non-corresponding entries are zero (sample a few)
    for _ in range(1000):
        i_in = random.randint(0, nm - 1)
        j_in = random.randint(0, mp - 1)
        k_out = random.randint(0, pn - 1)
        
        i_A = i_in // m
        j_A = i_in % m
        j_B = j_in // p
        k_B = j_in % p
        i_C = k_out % n
        k_C = k_out // n
        
        # Should be nonzero only if j_A == j_B and i_A == i_C
        if j_A == j_B and i_A == i_C:
            expected = 1.0
        else:
            expected = 0.0
        
        actual = T[i_in, j_in, k_out]
        err = abs(actual - expected)
        if err > tol:
            errors += 1
            max_err = max(max_err, err)
    
    return errors == 0, max_err


def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    best_result = schoolbook_decomposition()
    
    # Try to find a lower-rank decomposition
    # We'll use a simple approach: start with schoolbook and try to find
    # linear dependencies among the rank-1 terms
    
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    if remaining() > 10:
        # Build the standard tensor and try to find a minimal rank decomposition
        T = np.zeros((nm, mp, pn), dtype=float)
        for i_A in range(n):
            for j in range(m):
                for k in range(p):
                    i_input = i_A * m + j
                    j_input = j * p + k
                    k_output = k * n + i_A
                    T[i_input, j_input, k_output] = 1.0
        
        # The tensor has rank at most 42 (schoolbook).
        # Try to see if we can find a lower rank decomposition.
        # For small tensors, we can try random linear combinations.
        
        # Strategy: pick random subsets of size r from the 42 schoolbook terms
        # and check if they can be combined to form a valid decomposition.
        # This is not guaranteed to work, but let's try a few random searches.
        
        # Actually, a better approach for small problems: use the fact that
        # the tensor is a matrix multiplication tensor, and known bounds exist.
        # For 2x3 * 3x7, the tensor rank is at most 42 (trivial) and at least
        # ceil(14/2) = 7 * something... The actual rank is unknown but likely
        # between 20 and 42.
        
        # Let's try a random search: generate random rank-r decompositions and
        # verify them. Start with r = 30 and work down.
        
        for target_rank in range(35, 20, -1):
            if remaining() < 5:
                break
            
            for attempt in range(5):
                if remaining() < 2:
                    break
                
                # Generate a random decomposition of the given rank
                # U: (nm, rank), V: (mp, rank), W: (pn, rank)
                U_rand = np.random.randn(nm, target_rank)
                V_rand = np.random.randn(mp, target_rank)
                W_rand = np.random.randn(pn, target_rank)
                
                # Normalize to have entries near 0, 0.5, 1
                # Scale so that the product sums to approximately 1
                U_rand = np.clip(U_rand, -1, 1)
                V_rand = np.clip(V_rand, -1, 1)
                W_rand = np.clip(W_rand, -1, 1)
                
                # Check if this random decomposition is close to the target
                T_computed = np.zeros((nm, mp, pn), dtype=float)
                for t in range(target_rank):
                    T_computed += np.einsum('a,b,c->abc', 
                                            U_rand[:, t], V_rand[:, t], W_rand[:, t])
                
                # Compare with target
                diff = np.abs(T_computed - T)
                if np.max(diff) < 1e-6:
                    # Found a valid decomposition!
                    # Convert to integer/half-integer form if possible
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
            else:
                continue
            break
    
    # Verify the result
    valid, max_err = verify_decomposition(best_result["U"], best_result["V"], best_result["W"])
    if not valid:
        # Fallback to schoolbook
        best_result = schoolbook_decomposition()
    
    return best_result
