"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    - Start with the known-correct schoolbook rank-56 decomposition.
    - Use a small deterministic search for rank reduction by checking
      whether certain tensor coordinates can be eliminated via linear
      dependencies among the existing rank-1 terms.
    - The search is constrained to preserve exact correctness (verified
      by tensor reconstruction), and we keep the best valid decomposition
      found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    # --- Build the schoolbook rank-56 decomposition ---
    def build_schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # --- Verify correctness of a decomposition ---
    def verify(U, V, W):
        """Check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the mult tensor."""
        # Build the target tensor T[a,b,c] = 1 if a*b... 
        # Actually the matrix multiplication tensor: T[i,j,k] = 1 
        # The mult tensor is: for each (i,j,k), T = 1, and it's outer products.
        # We check: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] should be 1
        # for the (i,j,k) that participate.
        # In schoolbook, each (i,j,k) has exactly one term.
        # General check: build T as a 3-index array
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0  # mult tensor is all-ones for this shape? 
                    # Actually no. The matrix multiplication tensor M satisfies:
                    # (C = A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
                    # So the tensor has entries T[i,j,k] = 1 for all i,j,k.
                    # That's the structure: C[i,k] = sum_j A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Now check decomposition
        # T[i,j,k] = sum_r U[i*m+j... wait, indices need care.
        # U has shape (n*m, rank), indexed by row = i*m + j
        # V has shape (m*p, rank), indexed by row = j*p + k
        # W has shape (p*n, rank), indexed by row = k*n + i
        # So T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > 1e-12:
                        return False
        return True

    # --- Try to reduce rank by finding redundant columns ---
    # Use a simple heuristic: try removing each column and checking if
    # the remaining set still spans the tensor. This is expensive, so
    # we do a limited random search.
    
    if time.time() < deadline:
        # Try a greedy removal: for each column, check if removing it
        # still gives a valid decomposition.
        # This is O(rank^2 * n*m*p) which is fine for rank=56.
        current_rank = best_rank
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        
        # Precompute tensor values for faster verification
        def tensor_check_from_factors(Uf, Vf, Wf):
            """Fast check using vectorized operations."""
            # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # We can do this by building the full tensor via einsum
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # We need to index U by (i,j), V by (j,k), W by (k,i)
            # Let's use a direct loop since n*m*p = 56 is small
            n_local, m_local, p_local = n, m, p
            for i in range(n_local):
                for j in range(m_local):
                    u_idx = i * m_local + j
                    for k in range(p_local):
                        v_idx = j * p_local + k
                        w_idx = k * n_local + i
                        s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        # Greedy: try removing columns one by one
        removed_cols = []
        col_indices = list(range(current_rank))
        
        # Try removing in a fixed order for determinism
        for try_remove in sorted(col_indices):
            if time.time() > deadline - 5:
                break
            if try_remove in removed_cols:
                continue
            
            # Check if we can remove this column
            new_indices = [c for c in range(current_rank) if c != try_remove]
            if len(new_indices) == 0:
                break
            
            test_U = current_U[:, new_indices]
            test_V = current_V[:, new_indices]
            test_W = current_W[:, new_indices]
            
            if tensor_check_from_factors(test_U, test_V, test_W):
                # We can remove this column
                removed_cols.append(try_remove)
                current_rank = len(new_indices)
                current_U = test_U
                current_V = test_V
                current_W = test_W
                
                # Update best
                best_rank = current_rank
                best_U = current_U
                best_V = current_V
                best_W = current_W
                
                # Re-sort remaining columns
                col_indices = list(range(current_rank))
        
        # Try a second pass after first removal round
        if time.time() < deadline - 10:
            removed_cols = []
            col_indices = list(range(current_rank))
            for try_remove in sorted(col_indices):
                if time.time() > deadline - 5:
                    break
                if try_remove in removed_cols:
                    continue
                
                new_indices = [c for c in range(current_rank) if c != try_remove]
                if len(new_indices) == 0:
                    break
                
                test_U = current_U[:, new_indices]
                test_V = current_V[:, new_indices]
                test_W = current_W[:, new_indices]
                
                if tensor_check_from_factors(test_U, test_V, test_W):
                    removed_cols.append(try_remove)
                    current_rank = len(new_indices)
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    
                    best_rank = current_rank
                    best_U = current_U
                    best_V = current_V
                    best_W = current_W
                    
                    col_indices = list(range(current_rank))

    # Return in list-of-lists format
    best_U = best_U[:best_U.shape[0], :best_U.shape[1]]
    best_V = best_V[:best_V.shape[0], :best_V.shape[1]]
    best_W = best_W[:best_W.shape[0], :best_W.shape[1]]

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    # Convert to integer if possible
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
