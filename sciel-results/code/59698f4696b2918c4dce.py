import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135

    # Strassen 2x2 decomposition (rank 7)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        return U, V, W

    # 2-level Strassen for 4x4*4x4 (rank 49)
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                for i1 in range(2):
                    for i2 in range(2):
                        row = i1 * 2 + i2
                        U[row, col] = U2[i1, t1] * U2[i2, t2]
                        V[row, col] = V2[i1, t1] * V2[i2, t2]
                        W[row, col] = W2[i1, t1] * W2[i2, t2]
        return U, V, W

    # Rectangular 4x4x1 (rank 4)
    def rect_4x4x1():
        rank = 4
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        for col in range(4):
            for i in range(4):
                U[i * 4 + col, col] = 1.0
                W[i, col] = 1.0
            V[col * 5, col] = 1.0
        return U, V, W

    # Build initial decomposition: rank 53
    U44, V44, W44 = strassen_4x4()
    U41, V41, W41 = rect_4x4x1()
    
    rank_total = 53
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U44
    U[:, 49:] = U41
    V[:16, :49] = V44
    V[:, 49:] = V41
    W[:16, :49] = W44
    W[:, 49:] = W41

    # Verify
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if j_a != j_b or i_a != i_c or k_b != k_c:
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Efficient rank reduction using tensor unfolding + SVD
    if time.time() < deadline - 20:
        # Build full tensor
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank_total):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Try to find linear dependencies among columns
        # Unfold tensor along each mode and check rank
        # If columns are linearly dependent in a certain sense, we can merge
        
        # Strategy: For each pair of columns, check if their tensor contributions
        # are linearly dependent. If so, we can potentially remove one.
        
        # More efficient: use the fact that if sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T,
        # then we need the minimum number of terms.
        
        # Try greedy removal with fast verification using precomputed contributions
        active = list(range(rank_total))
        
        # Precompute contributions
        contribs = []
        for r in range(rank_total):
            c = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            contribs.append(c)
        
        # Sort by norm (remove least important first)
        norms = [np.linalg.norm(c) for c in contribs]
        order = sorted(range(rank_total), key=lambda x: norms[x])
        
        T_current = T.copy()
        
        for r_idx in order:
            if time.time() > deadline - 25:
                break
            if r_idx not in active:
                continue
            
            # Try removing this column
            T_trial = T_current - contribs[r_idx]
            
            # Check if T_trial can still be represented with remaining columns
            # This is hard in general, but for our structured decomposition,
            # we can check if the residual is zero (shouldn't be) or if it's
            # in the span of remaining contributions.
            
            # Simpler check: just verify the decomposition directly
            if len(active) <= 1:
                break
            
            new_active = [x for x in active if x != r_idx]
            
            # Quick check: rebuild tensor from remaining columns and compare
            T_rebuilt = np.zeros((n*m, m*p, p*n))
            for r in new_active:
                T_rebuilt += contribs[r]
            
            if np.allclose(T, T_rebuilt, atol=1e-9):
                active = new_active
                T_current = T_rebuilt
        
        if len(active) < rank_total:
            U = U[:, active]
            V = V[:, active]
            W = W[:, active]
            rank_total = len(active)

    # Try additional reduction: look for columns that are scalar multiples
    # or linear combinations in the factor matrices
    if time.time() < deadline - 15:
        # Check for duplicate or proportional columns
        keep = []
        for r in range(rank_total):
            if r in keep:
                continue
            is_dup = False
            for s in keep:
                # Check if column r is proportional to column s
                # U[:,r] = alpha * U[:,s] AND V[:,r] = alpha * V[:,s] AND W[:,r] = alpha * W[:,s]
                # Actually for tensor decomposition, we need:
                # U[:,r]*V[:,r]*W[:,r] proportional to U[:,s]*V[:,s]*W[:,s]
                cr = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                cs = np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                if np.linalg.norm(cs) > 1e-12:
                    alpha = np.dot(cr.flatten(), cs.flatten()) / np.dot(cs.flatten(), cs.flatten())
                    if np.allclose(cr, alpha * cs, atol=1e-9):
                        is_dup = True
                        break
            if not is_dup:
                keep.append(r)
        
        if len(keep) < rank_total:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank_total = len(keep)

    # Final verification
    if not verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    else:
        rank = rank_total

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
