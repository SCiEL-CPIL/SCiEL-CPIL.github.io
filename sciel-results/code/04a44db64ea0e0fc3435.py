"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_2x2x3_strassen(n, m, p, j0, j1, k0, k1, k2, U, V, W, col):
    """
    Build a 2x2x3 Strassen-like decomposition for rows j0,j1 and cols k0,k1,k2.
    Uses 7 products to compute:
      C[i, k0] = A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
      C[i, k1] = A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]  
      C[i, k2] = A[i,j0]*B[j0,k2] + A[i,j1]*B[j1,k2]
    for i=0,1 (the two rows of A)
    
    Products:
      P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
      P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
      P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
      P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
      P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
      P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
      P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    
    For k2, we need additional products since 2x2x3 > 2x2x2.
    Actually for 2x2x3, we can use:
      P8 = A[0,j0]*B[j0,k2]
      P9 = A[0,j1]*B[j1,k2]
      P10 = A[1,j0]*B[j0,k2]
      P11 = A[1,j1]*B[j1,k2]
    That gives 11 total, not better than 2*3=6 schoolbook for the k2 part + 7 for k0,k1.
    
    Let me think differently. For 2x2x3:
    We can split as 2x2x2 + 2x2x1.
    2x2x2 uses 7 products.
    2x2x1 uses 2 products (A[0,j0]*B[j0,k2] and A[0,j1]*B[j1,k2] for row 0, 
    A[1,j0]*B[j0,k2] and A[1,j1]*B[j1,k2] for row 1) = 4 products.
    Total: 7 + 4 = 11. But schoolbook is 2*2*3 = 12. So we save 1.
    
    Alternatively, maybe we can do better. Let's just use the 7-product 2x2x2 
    for k0,k1 and 4 products for k2.
    """
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    # For k2: 4 schoolbook products
    for i in range(2):
        for j in [j0, j1]:
            U[i*m + j][col] += 1
            V[j*p + k2][col] += 1
            W[k2*n + i][col] += 1
            col += 1
    
    return col

def _build_2x1x3_strassen(n, m, p, j, k0, k1, k2, U, V, W, col):
    """
    Build a 2x1x3 decomposition for row j and cols k0,k1,k2.
    This is just 2*3 = 6 schoolbook products, but we can try to do better.
    Actually 2x1x3 has tensor rank at most 6, and likely exactly 6.
    """
    for i in range(2):
        for k in [k0, k1, k2]:
            U[i*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + i][col] += 1
            col += 1
    return col

def _try_eliminate_terms(U, V, W, n, m, p, rank):
    """
    Try to eliminate redundant terms by checking if removing term r
    still gives the correct tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor contribution matrix
    # For each (i, l, k), the contribution from term r is U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    
    # We want to find if any term r can be removed.
    # For term r, its contribution vector is:
    #   c_r[i*m+l, l*p+k, k*n+i] = U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    # But the indices are coupled: (i,l,k) -> (i*m+l, l*p+k, k*n+i)
    
    # Let's build a matrix where each row is a (i,l,k) triple and each column is a term r.
    # Entry (triple, r) = U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r]
    
    triples = []
    for i in range(n):
        for l in range(m):
            for k in range(p):
                triples.append((i, l, k))
    
    # Build contribution matrix
    contrib = np.zeros((len(triples), rank))
    for idx, (i, l, k) in enumerate(triples):
        for r in range(rank):
            contrib[idx, r] = U_arr[i*m+l, r] * V_arr[l*p+k, r] * W_arr[k*n+i, r]
    
    # The target is all 1s
    target = np.ones(len(triples))
    
    # Check if any column can be removed: solve (contrib without col r) x = target
    # and check if the solution has small norm or if the residual is small
    
    # Actually, we need the sum to equal target, so we need:
    # sum_r contrib[:, r] * 1 = target (since each term is weighted by 1 in the product)
    # Wait, that's not right. The decomposition is:
    # sum_r U[i*m+l,r] * V[l*p+k,r] * W[k*n+i,r] = 1
    # So each term contributes exactly U*V*W, and we need the sum to be 1.
    # So we need: contrib @ ones(rank) = target
    
    # To check if we can remove term r, we need to find weights w_s (s != r) such that:
    # sum_{s != r} contrib[:, s] * w_s = target
    # This is a linear system: contrib[:, :r] @ w[:r] + contrib[:, r+1:] @ w[r+1:] = target
    
    # But the weights must be such that the decomposition still works.
    # Actually, in a tensor decomposition, the weights are absorbed into U, V, W.
    # So we can't just change weights; we need to find new U', V', W' of rank rank-1.
    
    # A simpler check: if term r has all zeros in its contribution vector, it's redundant.
    # Or if its contribution vector is in the span of others.
    
    # Let's check if the contribution vector of term r is in the span of others.
    for r in range(rank):
        if time.time() > 130:
            break
        # Check if contrib[:, r] is in span of {contrib[:, s] : s != r}
        other_cols = np.delete(contrib, r, axis=1)
        if other_cols.shape[1] == 0:
            continue
        # Check rank
        rank_other = np.linalg.matrix_rank(other_cols, tol=1e-10)
        # Check if adding contrib[:, r] increases the rank
        combined = np.column_stack([other_cols, contrib[:, r]])
        rank_combined = np.linalg.matrix_rank(combined, tol=1e-10)
        if rank_combined == rank_other:
            # Term r is redundant! Remove it.
            print(f"Term {r} is redundant, removing...")
            U_new = [row[:r] + row[r+1:] for row in U]
            V_new = [row[:r] + row[r+1:] for row in V]
            W_new = [row[:r] + row[r+1:] for row in W]
            return U_new, V_new, W_new, rank - 1
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Strategy: Use 2x2x3 blocks for (j0,j1) pairs and k0,k1,k2 triples
    # j pairs: (0,1), (2,3), and single j=4
    # k triples: (0,1,2), (3,4,5)
    
    # For each (j0,j1) pair and k triple (k0,k1,k2):
    # Use 7 products for the 2x2x2 part (k0,k1) + 4 products for k2 = 11 total
    # For j=4: 2x1x3 = 6 products
    
    # Total: 2 j-pairs * 2 k-triples * 11 + 1 j-single * 2 k-triples * 6
    # = 2*2*11 + 1*2*6 = 44 + 12 = 56. Worse than 54!
    
    # Let me try a different partition:
    # j pairs: (0,1), (2,3), (4) 
    # k pairs: (0,1), (2,3), (4,5)
    # For (j0,j1) and (k0,k1): 7 products (2x2x2 Strassen)
    # For j=4 and (k0,k1): 4 products (2x1x2 schoolbook)
    # Total: 2 j-pairs * 3 k-pairs * 7 + 1 j-single * 3 k-pairs * 4
    # = 2*3*7 + 1*3*4 = 42 + 12 = 54. Same as current!
    
    # So the current 54 is from this partition. Let's try to find redundancies.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # Verify initial decomposition
    if not _verify_decomp(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Try to eliminate redundant terms
    for attempt in range(10):
        if time.time() > _deadline - 10:
            break
        U, V, W, rank = _try_eliminate_terms(U, V, W, n, m, p, rank)
        if not _verify_decomp(U, V, W, n, m, p, rank):
            # Restore previous
            break
        if rank <= 50:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
