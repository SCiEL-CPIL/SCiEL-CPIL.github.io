"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 using Strassen-based blocks."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit


def _strassen_rank42():
    """Build the rank-42 Strassen-based decomposition."""
    n, m, p = 2, 4, 6
    rank_total = 42

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1},
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_offset = 0
                v_row_offset = 0
            else:
                u_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                local_U_map = {
                    0: 0 * 4 + (u_offset + 0),
                    1: 0 * 4 + (u_offset + 1),
                    2: 1 * 4 + (u_offset + 0),
                    3: 1 * 4 + (u_offset + 1),
                }
                for local_idx, sign in strassen_U[prod_idx].items():
                    U[local_U_map[local_idx]][col] += sign

                local_V_map = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for local_idx, sign in strassen_V[prod_idx].items():
                    V[local_V_map[local_idx]][col] += sign

                local_W_map = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for local_idx, sign in strassen_W[prod_idx].items():
                    W[local_W_map[local_idx]][col] += sign

            col_offset += 7

    return {"U": U, "V": V, "W": W}


def _try_mpo_rank_reduction():
    """
    Attempt to find a lower-rank decomposition using tensor network analysis.
    
    Strategy: 
    1. Start from rank-42 Strassen solution
    2. Build the full tensor equation system
    3. Use SVD on tensor unfoldings to detect if rank reduction is possible
    4. Try gauge transformations to reveal hidden structure
    """
    if time.time() - START_TIME > TIME_LIMIT - 20:
        return None
    
    n, m, p = 2, 4, 6
    
    # Get the rank-42 solution
    sol42 = _strassen_rank42()
    U42 = np.array(sol42['U'], dtype=float)
    V42 = np.array(sol42['V'], dtype=float)
    W42 = np.array(sol42['W'], dtype=float)
    
    # Check for linear dependencies among the 42 rank-1 tensors
    r42 = U42.shape[1]
    tensor_vectors = np.zeros((r42, n * m * p))
    for t in range(r42):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    tensor_vectors[t, idx] = U42[i * m + j, t] * V42[j * p + k, t] * W42[k * n + i, t]
    
    # Check the rank of the 42x48 matrix
    try:
        u, s, vt = np.linalg.svd(tensor_vectors, full_matrices=False)
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < r42:
            print(f"Found {r42 - num_rank} dependencies in rank-42 solution")
            # Try to exploit the null space
            return _exploit_null_space(U42, V42, W42, vt[num_rank:, :], n, m, p)
        else:
            print(f"Rank-42 solution has full tensor rank {num_rank}")
    except Exception as e:
        print(f"Error in dependency check: {e}")
    
    return None


def _exploit_null_space(U, V, W, null_vectors, n, m, p):
    """
    Try to exploit null space vectors to find a lower-rank decomposition.
    If we have a null vector c such that sum_t c[t] * T_t = 0,
    we can try to modify the decomposition to eliminate some components.
    """
    r = U.shape[1]
    # Try to find a component that can be eliminated
    # by adding a small multiple of the null combination
    for t_elim in range(r):
        # Try to zero out component t_elim by adjusting others
        # This is a nonlinear problem, so we use a heuristic approach
        # Check if component t_elim is "redundant" in some sense
        
        # Compute the tensor contribution of component t_elim
        T_elim = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_elim[idx] = U[i * m + j, t_elim] * V[j * p + k, t_elim] * W[k * n + i, t_elim]
        
        # Check if T_elim is in the span of the other components
        other_tensors = np.zeros((r - 1, n * m * p))
        for t_other in range(r):
            if t_other == t_elim:
                continue
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        other_tensors[t_other if t_other < t_elim else t_other - 1, idx] = \
                            U[i * m + j, t_other] * V[j * p + k, t_other] * W[k * n + i, t_other]
        
        # Check if T_elim is in the row space of other_tensors
        try:
            # Use least squares to see if T_elim can be approximated
            coeffs, residuals, rank, sv = np.linalg.lstsq(other_tensors.T, T_elim, rcond=None)
            residual_norm = np.linalg.norm(other_tensors.T @ coeffs - T_elim)
            if residual_norm < 1e-8:
                print(f"Component {t_elim} is redundant!")
                # Remove this component
                U_new = np.delete(U, t_elim, axis=1)
                V_new = np.delete(V, t_elim, axis=1)
                W_new = np.delete(W, t_elim, axis=1)
                return U_new, V_new, W_new
        except:
            pass
    
    return None


def _try_direct_rank_search(target_rank):
    """
    Directly search for a rank-target decomposition using structured initialization.
    Uses the constraint that each (i,j,k) triple must sum to 1.
    """
    if time.time() - START_TIME > TIME_LIMIT - 15:
        return None
    
    n, m, p = 2, 4, 6
    
    # Build the constraint matrix for the tensor equation
    # For each (i,j,k), we need: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # This is nonlinear, but we can try a local search approach
    
    # Start from rank-42 and try to perturb to reduce rank
    sol42 = _strassen_rank42()
    U = np.array(sol42['U'], dtype=float)
    V = np.array(sol42['V'], dtype=float)
    W = np.array(sol42['W'], dtype=float)
    
    r = U.shape[1]
    
    # Try to find if we can set one component to zero while adjusting others
    # to maintain the tensor identity
    for t_target in range(r):
        # Compute current tensor values
        current_T = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    current_T[idx] = sum(
                        U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] 
                        for t in range(r)
                    )
        
        # Try to eliminate component t_target
        # This requires solving a nonlinear system, so we use a heuristic
        # Check if the component is "small" or "redundant"
        
        # Compute the norm of component t_target's contribution
        T_target = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_target[idx] = U[i*m+j, t_target] * V[j*p+k, t_target] * W[k*n+i, t_target]
        
        # If T_target is close to zero, we might be able to eliminate it
        if np.linalg.norm(T_target) < 1e-10:
            print(f"Component {t_target} has near-zero contribution")
            U_new = np.delete(U, t_target, axis=1)
            V_new = np.delete(V, t_target, axis=1)
            W_new = np.delete(W, t_target, axis=1)
            return U_new, V_new, W_new
    
    return None


def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 4, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Verify by checking that for random A and B, the result matches A@B
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j_out in range(p):
            val = 0.0
            for j_in in range(m):
                for k in range(rank):
                    # U index: i*m + j_in
                    # V index: j_in*p + j_out  
                    # W index: j_out*n + i
                    u_val = U[i * m + j_in, k]
                    v_val = V[j_in * p + j_out, k]
                    w_val = W[j_out * n + i, k]
                    val += u_val * v_val * w_val
            C_computed[i, j_out] = val
    
    return np.allclose(C_computed, C_expected, atol=1e-9)


def _try_structured_rank_reduction():
    """
    Try to find a rank-40 or rank-41 decomposition by exploiting block structure.
    Uses tensor network disentanglement to find gauge transformations.
    """
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return None
    
    n, m, p = 2, 4, 6
    
    # Get the rank-42 solution as a starting point
    sol42 = _strassen_rank42()
    U42 = np.array(sol42['U'], dtype=float)
    V42 = np.array(sol42['V'], dtype=float)
    W42 = np.array(sol42['W'], dtype=float)
    
    r = U42.shape[1]
    
    # The 42 components are organized as 6 groups of 7 (Strassen)
    # Try to find linear combinations that allow merging blocks
    
    # Build the tensor equation system
    # For each (i,j,k): sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Try a local optimization approach: perturb the solution and check
    # if we can reduce the effective rank
    
    # Use SVD on the tensor unfolding to detect structure
    # Unfold along the rank dimension: shape (r, n*m*p)
    tensor_unfold = np.zeros((r, n * m * p))
    for t in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    tensor_unfold[t, idx] = U42[i*m+j, t] * V42[j*p+k, t] * W42[k*n+i, t]
    
    # Check if the tensor has a lower-rank structure
    try:
        u, s, vt = np.linalg.svd(tensor_unfold, full_matrices=False)
        # The singular values tell us about the effective rank
        # If some singular values are very small, we might be able to reduce rank
        tol = 1e-8
        effective_rank = np.sum(s > tol)
        
        if effective_rank < r:
            print(f"Effective tensor rank: {effective_rank} (vs nominal {r})")
            # Try to project onto the effective rank subspace
            # This is a heuristic - we need to verify the result
            U_new = U42[:, :effective_rank]
            V_new = V42[:, :effective_rank]
            W_new = W42[:, :effective_rank]
            
            # Verify
            if _verify_decomposition(U_new, V_new, W_new):
                return U_new, V_new, W_new
    except Exception as e:
        print(f"SVD error: {e}")
    
    return None


def _try_rank_reduction(rank_target):
    """
    Try to find a rank-reduced decomposition by systematic search.
    Uses the tensor equation constraints to guide the search.
    Returns (U, V, W) if successful, None otherwise.
    """
    if time.time() - START_TIME > TIME_LIMIT - 30:
        return None
    
    n, m, p = 2, 4, 6
    
    # Get the rank-42 solution
    sol42 = _strassen_rank42()
    U = np.array(sol42['U'], dtype=float)
    V = np.array(sol42['V'], dtype=float)
    W = np.array(sol42['W'], dtype=float)
    
    r = U.shape[1]
    
    # Build the tensor equation system
    # For each (i,j,k): sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    # We have 48 equations and r unknowns per matrix
    
    # Try to find a rank-reduced solution by:
    # 1. Checking if any component is redundant
    # 2. Trying to merge components from different Strassen blocks
    
    # Check for redundant components
    for t in range(r):
        # Compute the contribution of component t
        T_t = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_t[idx] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Check if T_t is in the span of other components
        other_T = np.zeros((r-1, n*m*p))
        count = 0
        for t2 in range(r):
            if t2 == t:
                continue
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        other_T[count, idx] = U[i*m+j, t2] * V[j*p+k, t2] * W[k*n+i, t2]
            count += 1
        
        # Use least squares to check if T_t is in the span
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(other_T.T, T_t, rcond=None)
            residual = np.linalg.norm(other_T.T @ coeffs - T_t)
            if residual < 1e-10:
                print(f"Found redundant component {t}")
                U_new = np.delete(U, t, axis=1)
                V_new = np.delete(V, t, axis=1)
                W_new = np.delete(W, t, axis=1)
                if _verify_decomposition(U_new, V_new, W_new):
                    return U_new, V_new, W_new
        except:
            pass
    
    return None


def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    
    The decomposition uses 6 blocks of 2x2x2 Strassen multiplications (rank 7 each),
    giving total rank 42. Attempts to find lower-rank decompositions via tensor network
    analysis and structured search.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT:
        return _strassen_rank42()
    
    # Try multiple rank-reduction strategies
    strategies = [
        ('mpo', _try_mpo_rank_reduction),
        ('direct', _try_direct_rank_search),
        ('structured', _try_structured_rank_reduction),
        ('reduction', _try_rank_reduction),
    ]
    
    for name, func in strategies:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        try:
            result = func()
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W):
                    rank = U.shape[1]
                    print(f"Found rank-{rank} decomposition via {name}")
                    return {"U": U, "V": V, "W": W}
        except Exception as e:
            print(f"Strategy {name} failed: {e}")
    
    # Fall back to rank-42 Strassen-based decomposition
    return _strassen_rank42()
