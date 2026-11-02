import time
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Uses exterior algebra / Grassmann-Cayley pruning to search for
    a rank lower than the schoolbook rank of 24.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = 2, 3, 4
    rank_schoolbook = n * m * p  # 24
    
    # 1. Construct the known rank-24 solution (Schoolbook)
    U, V, W = _schoolbook(n, m, p)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # 2. Attempt structured search for rank 23 using exterior algebra pruning
    if time.time() < deadline - 15.0:
        result_lower = _try_rank_exterior(n, m, p, best_rank - 1, deadline)
        if result_lower is not None:
            best_result = result_lower
            best_rank = best_rank - 1
    
    # 3. If still at 24, try a more aggressive local search
    if best_rank == rank_schoolbook and time.time() < deadline - 5.0:
        result_local = _try_rank_local_polish(n, m, p, best_rank, deadline)
        if result_local is not None:
            best_result = result_local
            best_rank = best_rank - 1
    
    return best_result

def _schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
    
    return U, V, W

def _try_rank_exterior(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition of rank 'target_rank' using exterior algebra
    and Grassmann-Cayley pruning.
    
    Strategy: Start from the schoolbook rank-24 decomposition and try to merge
    pairs of rank-1 tensors by checking compatibility in the exterior algebra.
    """
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # Build the target multiplication tensor T as a flat vector
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Get schoolbook decomposition as numpy arrays
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_sb = np.array(U_sb, dtype=np.float64)
    V_sb = np.array(V_sb, dtype=np.float64)
    W_sb = np.array(W_sb, dtype=np.float64)
    
    # Try to merge pairs of rank-1 tensors
    # For each pair (i, j), check if U[:,i]⊗V[:,j]⊗W[:,k] and U[:,i']⊗V[:,j']⊗W[:,k']
    # can be combined into a single rank-1 tensor
    
    # Represent each rank-1 tensor as a vector in the tensor product space
    # Two rank-1 tensors can be merged if their "directions" are compatible
    
    # Use a heuristic: try swapping indices in the schoolbook decomposition
    # and check if we can reduce the rank by 1
    
    # Approach: For each pair of columns, try to see if they can be merged
    # by checking if the 2x2 minors of the coefficient matrix vanish
    
    max_pairs_to_check = 10000
    pairs_checked = 0
    
    for i in range(target_rank + 1):
        if time.time() > deadline:
            return None
        for j in range(i + 1, target_rank + 1):
            if time.time() > deadline:
                return None
            if pairs_checked >= max_pairs_to_check:
                break
            
            pairs_checked += 1
            
            # Try to merge columns i and j by creating a new column
            # that is a linear combination
            # Check if the wedge product vanishes
            
            # Get the two rank-1 tensors
            u1 = U_sb[:, i]
            v1 = V_sb[:, i]
            w1 = W_sb[:, i]
            u2 = U_sb[:, j]
            v2 = V_sb[:, j]
            w2 = W_sb[:, j]
            
            # Check if these two rank-1 tensors are "compatible" for merging
            # by testing if there exist scalars alpha, beta such that
            # (alpha*u1 + beta*u2) ⊗ (alpha*v1 + beta*v2) ⊗ (alpha*w1 + beta*w2)
            # equals u1⊗v1⊗w1 + u2⊗v2⊗w2 (up to scaling)
            
            # This is a nonlinear constraint, so use a numerical test
            # Sample a few points to check compatibility
            
            # For a quick test: check if the cross-ratio of the vectors is consistent
            # Use the exterior algebra: the wedge product u1∧u2, v1∧v2, w1∧w2
            # should satisfy Plücker relations if mergeable
            
            # Simplified test: check if the 2x2 minors are proportional
            u1u2 = np.outer(u1, u2) - np.outer(u2, u1)  # antisymmetric
            v1v2 = np.outer(v1, v2) - np.outer(v2, v1)
            w1w2 = np.outer(w1, w2) - np.outer(w2, w1)
            
            # If any of these is zero, the vectors are linearly dependent
            # which is a necessary (not sufficient) condition for merging
            if (np.linalg.norm(u1u2) < 1e-10 or 
                np.linalg.norm(v1v2) < 1e-10 or 
                np.linalg.norm(w1w2) < 1e-10):
                # Try to actually merge by solving for the combined tensor
                merged = _try_merge_pair(U_sb, V_sb, W_sb, i, j, T, nm, mp, pn)
                if merged is not None:
                    return merged
    
    return None

def _try_merge_pair(U_sb, V_sb, W_sb, i, j, T, nm, mp, pn):
    """
    Try to merge columns i and j of the schoolbook decomposition.
    Returns a new decomposition of rank 23 if successful, else None.
    """
    # Create new U, V, W with columns i and j merged
    # The new column k will be a linear combination
    
    # Try a few linear combinations
    for alpha in [0.5, -0.5, 1.0, -1.0]:
        beta = 1.0 - alpha
        
        u_new = alpha * U_sb[:, i] + beta * U_sb[:, j]
        v_new = alpha * V_sb[:, i] + beta * V_sb[:, j]
        w_new = alpha * W_sb[:, i] + beta * W_sb[:, j]
        
        # Check if this new column plus all other columns (except i, j)
        # still decomposes T
        U_test = np.column_stack([
            U_sb[:, :i], U_sb[:, i+1:j], U_sb[:, j+1:],
            u_new.reshape(-1, 1)
        ])
        V_test = np.column_stack([
            V_sb[:, :i], V_sb[:, i+1:j], V_sb[:, j+1:],
            v_new.reshape(-1, 1)
        ])
        W_test = np.column_stack([
            W_sb[:, :i], W_sb[:, i+1:j], W_sb[:, j+1:],
            w_new.reshape(-1, 1)
        ])
        
        # Verify the decomposition
        tensor = np.einsum('ac,bc,dc->abd', U_test, V_test, W_test)
        if np.allclose(tensor, T, atol=1e-9):
            # Snap to half-integers
            U_snap = np.round(U_test * 2) / 2
            V_snap = np.round(V_test * 2) / 2
            W_snap = np.round(W_test * 2) / 2
            
            tensor_snap = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
            if np.allclose(tensor_snap, T, atol=1e-9):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    return None

def _try_rank_local_polish(n, m, p, current_rank, deadline):
    """
    Try a local polish around the schoolbook solution to see if we can
    reduce the rank by 1 using a more systematic search.
    """
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    
    # Build the target multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_sb = np.array(U_sb, dtype=np.float64)
    V_sb = np.array(V_sb, dtype=np.float64)
    W_sb = np.array(W_sb, dtype=np.float64)
    
    # Try a structured perturbation: modify one column at a time
    # and check if we can drop another column
    
    target_rank = current_rank - 1
    
    for col_to_drop in range(current_rank):
        if time.time() > deadline:
            return None
        
        # Remove column col_to_drop and try to adjust the remaining columns
        # to compensate
        
        U_rem = np.delete(U_sb, col_to_drop, axis=1)
        V_rem = np.delete(V_sb, col_to_drop, axis=1)
        W_rem = np.delete(W_sb, col_to_drop, axis=1)
        
        # The current decomposition with col_to_drop removed is missing
        # the rank-1 tensor that was in that column
        # Try to absorb it into the remaining columns
        
        # Compute the residual
        tensor_current = np.einsum('ac,bc,dc->abd', U_rem, V_rem, W_rem)
        residual = T - tensor_current
        
        # The residual is a single rank-1 tensor (the one we dropped)
        # Try to express it as a linear combination of the remaining rank-1 tensors
        # by adjusting the coefficients
        
        # This is a linear system: find alpha_c such that
        # sum_c alpha_c * U_c ⊗ V_c ⊗ W_c = residual
        
        # Flatten the tensor equation
        # This is a large linear system, so use a least-squares approach
        
        # Build the design matrix
        # Each rank-1 tensor contributes to the flattened residual
        # The system is: sum_c alpha_c * vec(U_c ⊗ V_c ⊗ W_c) = vec(residual)
        
        # This is a 24x23 system (after dropping one column)
        # Try to solve it
        
        # Build the matrix where each column is the flattened rank-1 tensor
        design = np.zeros((nm * mp * pn, target_rank))
        for c in range(target_rank):
            u_c = U_rem[:, c]
            v_c = V_rem[:, c]
            w_c = W_rem[:, c]
            rank1 = np.einsum('a,b,c->abc', u_c, v_c, w_c)
            design[:, c] = rank1.flatten()
        
        residual_flat = residual.flatten()
        
        # Solve the least-squares problem
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(design, residual_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is close to exact
        design_alpha = design @ alpha
        if np.linalg.norm(design_alpha - residual_flat) < 1e-9:
            # We found a solution! Update the columns
            U_new = U_rem * alpha
            V_new = V_rem
            W_new = W_rem
            
            # Verify
            tensor_new = np.einsum('ac,bc,dc->abd', U_new, V_new, W_new)
            if np.allclose(tensor_new, T, atol=1e-9):
                # Snap to half-integers
                U_snap = np.round(U_new * 2) / 2
                V_snap = np.round(V_new * 2) / 2
                W_snap = np.round(W_new * 2) / 2
                
                tensor_snap = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
                if np.allclose(tensor_snap, T, atol=1e-9):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END
