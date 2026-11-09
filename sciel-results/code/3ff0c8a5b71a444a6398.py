"""Rewritten program for 3x5x5 matrix multiplication tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-minimal (or best found) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns a dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook decomposition as a guaranteed valid result
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try to find better decompositions
    # For 3x5x5, the schoolbook rank is 75
    
    # Strategy 1: Try Winograd-style transformations
    if time.time() < deadline:
        winograd_result = _try_winograd_rectangular(n, m, p)
        if winograd_result is not None:
            rank = len(winograd_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = winograd_result
    
    # Strategy 2: Try structured decompositions exploiting symmetry
    if time.time() < deadline:
        symmetric_result = _try_symmetric_decomposition(n, m, p)
        if symmetric_result is not None:
            rank = len(symmetric_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = symmetric_result
    
    # Strategy 3: Try rank reduction via linear combinations
    if time.time() < deadline:
        reduced_result = _try_rank_reduction(n, m, p, best_result, best_rank)
        if reduced_result is not None:
            rank = len(reduced_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = reduced_result
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    For each (i, j, k): 
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    This must hold for all valid i, j, k.
    """
    rank = len(U[0])
    # Build numpy arrays for efficient verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over all rank terms
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_winograd_rectangular(n, m, p):
    """
    Try Winograd-style transformation for rectangular matrix multiplication.
    
    For n=3, m=5, p=5, we can try to exploit structure.
    Winograd for matrix multiplication typically works best for square matrices
    where the block size divides the dimension evenly.
    
    For 3x5x5, we can try:
    - Decompose the 5x5 part using Winograd with block size 2 or 5
    - Handle the 3x5 part separately
    
    This is a heuristic approach; we'll try small rank reductions.
    """
    # For now, this is a placeholder that returns None
    # A full Winograd implementation would be complex
    # Let's try a simpler structured approach
    return None


def _try_symmetric_decomposition(n, m, p):
    """
    Try to exploit the symmetry between the two dimension-5 indices.
    
    Since m = p = 5, we can try decompositions that treat j and k symmetrically.
    Use a block-diagonal approach where the rank-3 matrix multiplication core
    is embedded within a larger structure using Kronecker products.
    """
    if m != p:
        return None
    
    # Try to construct a decomposition that exploits the m=p symmetry
    # by using permutation matrices and identity blocks
    
    # Approach: For each pair (j, k) where j=k, use a shared rank-1 term
    # For j != k, we still need separate terms but can potentially share structure
    
    # Let's try a structured approach:
    # Group the 25 (j,k) pairs into symmetric groups
    # j=k: 5 pairs (diagonal)
    # j<k: 10 pairs (upper triangle)
    # j>k: 10 pairs (lower triangle)
    
    # For the diagonal terms, we can potentially use fewer rank-1 terms
    # by exploiting that A[i,j]*B[j,j] only appears in C[i,j]
    
    # For off-diagonal terms, we need to be more careful
    
    # Let's try a rank reduction by combining terms that have the same
    # (i, j, k) pattern but different (i, k) indices
    
    # Actually, let's try a different approach: use the fact that for fixed i,
    # the computation C[i,k] = sum_j A[i,j] B[j,k] is a dot product of
    # the i-th row of A with the k-th column of B.
    
    # We can try to use a shared basis for the j-index across different i values
    
    # For n=3, m=5, p=5:
    # Each row of A (3 rows) needs to be multiplied by B (5x5)
    # This gives 3 * 25 = 75 scalar multiplications in schoolbook
    
    # Can we share any of these? Not obviously, since each A[i,j] is distinct.
    
    # Let's try a more sophisticated approach: use linear combinations
    # of the rank-1 terms to reduce the rank
    
    # Initialize with schoolbook and try to find linear dependencies
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Try to find a smaller rank by solving a system of equations
    # This is computationally expensive, so we'll try a heuristic approach
    
    # Heuristic: try random linear combinations of the rank-1 terms
    # and check if they still satisfy the tensor equation
    
    # For each subset of rank-1 terms, try to express one term as a
    # linear combination of the others
    
    # This is equivalent to finding the tensor rank, which is hard.
    # So we'll try a greedy approach: iteratively remove terms that
    # can be expressed as linear combinations of the remaining terms.
    
    # Convert to numpy for efficient computation
    U_arr = np.array(schoolbook['U'], dtype=float)
    V_arr = np.array(schoolbook['V'], dtype=float)
    W_arr = np.array(schoolbook['W'], dtype=float)
    
    rank = U_arr.shape[1]
    
    # Try to reduce rank by finding linear dependencies
    # For each column r, check if it can be expressed as a linear combination
    # of the other columns
    
    # This is equivalent to checking if the tensor rank is less than n*m*p
    # We'll try a simple heuristic: for each column, try to express it as
    # a linear combination of the other columns using least squares
    
    # But this is expensive, so we'll try a limited number of iterations
    
    max_iterations = 100
    current_rank = rank
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Try to remove one column at a time
        removed = False
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Try to express column r as a linear combination of the other columns
            # This is equivalent to checking if the tensor defined by column r
            # is in the span of the other tensors
            
            # Create the tensor for column r
            T_r = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_r[i, j, k] = current_U[i * m + j, r] * current_V[j * p + k, r] * current_W[k * n + i, r]
            
            # Try to express T_r as a linear combination of the other tensors
            # This is a linear system: sum_{r' != r} c_{r'} * T_{r'} = T_r
            
            # Build the linear system
            num_other_cols = current_rank - 1
            if num_other_cols <= 0:
                continue
            
            # The linear system is: sum_{r' != r} c_{r'} * T_{r'} = T_r
            # This is a system of n*m*p equations in num_other_cols unknowns
            
            # Build the matrix A and vector b
            A = np.zeros((n * m * p, num_other_cols))
            b = np.zeros(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b[idx] = T_r[i, j, k]
                        for r_prime in range(current_rank):
                            if r_prime == r:
                                continue
                            A[idx, r_prime if r_prime < r else r_prime - 1] = (
                                current_U[i * m + j, r_prime] * 
                                current_V[j * p + k, r_prime] * 
                                current_W[k * n + i, r_prime]
                            )
                        idx += 1
            
            # Solve the linear system
            try:
                coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is good (residual is small)
                if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-8:
                    # Check if the solution is exact
                    A_coeffs = A @ coeffs
                    if np.allclose(A_coeffs, b, atol=1e-8):
                        # We found a linear combination, so we can remove column r
                        # Update the other columns
                        for r_prime in range(current_rank):
                            if r_prime == r:
                                continue
                            coeff = coeffs[r_prime if r_prime < r else r_prime - 1]
                            current_U[:, r_prime] += coeff * current_U[:, r]
                            current_V[:, r_prime] += coeff * current_V[:, r]
                            current_W[:, r_prime] += coeff * current_W[:, r]
                        
                        # Remove column r
                        mask = [i for i in range(current_rank) if i != r]
                        current_U = current_U[:, mask]
                        current_V = current_V[:, mask]
                        current_W = current_W[:, mask]
                        current_rank -= 1
                        removed = True
                        break
            except np.linalg.LinAlgError:
                continue
        
        if not removed:
            break
    
    # Verify the result
    if current_rank < rank:
        result = {
            "U": current_U.tolist(),
            "V": current_V.tolist(),
            "W": current_W.tolist()
        }
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    return None


def _try_rank_reduction(n, m, p, base_result, base_rank):
    """
    Try to reduce the rank of the base decomposition by finding linear dependencies
    among the rank-1 terms.
    """
    # Placeholder - return None for now
    return None
# EVOLVE-BLOCK-END
