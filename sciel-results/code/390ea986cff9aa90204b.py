import numpy as np

def construct():
    n, m, p = 3, 5, 7
    
    # Cache for known ranks or computed decompositions
    # We will return U, V, W matrices
    # To make it efficient, we can just build the schoolbook for the whole thing
    # but the prompt asks for an improvement.
    
    # Let's use the recursive splitting to determine the structure.
    # However, implementing the actual tensor decomposition for arbitrary splits 
    # requires combining the U, V, W matrices of the sub-decompositions.
    
    # If we split dimension X, we can just concatenate the U, V, W blocks.
    # For example, if we split n into n1, n2:
    # U = [U1; U2] (vertical concat, but U has shape (n*m, rank). 
    # Wait, U index is (i, j). If we split i (n), the rows of U corresponding to i1 and i2 are separate.
    # V index is (j, k). Unchanged.
    # W index is (k, i). Rows for i1 and i2 are separate.
    # So U = vstack([U1, U2]), V = hstack([V1, V2]), W = vstack([W1, W2])?
    # No, V is shared? No, V depends on j, k. 
    # If we split n (i), the multiplication is:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # C is split into C1 (n1 x p) and C2 (n2 x p).
    # C1 = A1 B, C2 = A2 B.
    # Decomposition of C1 = A1 B: U1 (n1*m, r1), V1 (m*p, r1), W1 (p*n1, r1).
    # Decomposition of C2 = A2 B: U2 (n2*m, r2), V2 (m*p, r2), W2 (p*n2, r2).
    # The B matrix is the same in both!
    # So we can use the same V?
    # No, the rank decomposition is:
    # C = sum_t U[:, t] * V[:, t]^T * W[:, t]^T? No.
    # The tensor T[i, j, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t].
    # If we split i into i1, i2:
    # T1[i1, j, k] = sum_t U1[i1, j, t] V1[j, k, t] W1[k, i1, t].
    # T2[i2, j, k] = sum_t U2[i2, j, t] V2[j, k, t] W2[k, i2, t].
    # We can combine them into one tensor:
    # U_total[i, j, t] = U1[i1, j, t] if i in i1, U2[i2, j, t] if i in i2.
    # V_total[j, k, t] = V1[j, k, t] for t in 1..r1, V2[j, k, t] for t in r1+1..r1+r2.
    # W_total[k, i, t] = W1[k, i1, t] if i in i1, W2[k, i2, t] if i in i2.
    # This works. The rank is r1 + r2.
    
    # Similarly for splitting j (m) or k (p).
    
    # Let's define a function that returns U, V, W for given n, m, p.
    # We use a simple heuristic: try all splits and pick the one with minimal estimated rank.
    # For base cases (small n, m, p), use schoolbook or known values.
    
    def get_decomposition(n, m, p):
        # Base case: if any dimension is 1, rank is product of others? No.
        # If n=1, T[0, j, k] = A[0, j] B[j, k].
        # This is just the matrix B scaled by A.
        # Rank is m*p? No, A[0, j] is a vector.
        # T[0, j, k] = sum_t U[0, j, t] V[j, k, t] W[k, 0, t].
        # If we set W[k, 0, t] = 1, U[0, j, t] = A[0, j] delta_{t, j}? No.
        # Schoolbook: rank = m*p.
        
        # If n=1, m=1, p=1: rank 1.
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
            
        if n == 1:
            # T[0, j, k] = A[0, j] B[j, k]
            # We can use rank m*p.
            rank = m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for j in range(m):
                for k in range(p):
                    U[0 * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + 0, col] = 1.0
                    col += 1
            return U, V, W
            
        if m == 1:
            # T[i, 0, k] = A[i, 0] B[0, k]
            rank = n * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i * m + 0, col] = 1.0
                    V[0 * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
            return U, V, W
            
        if p == 1:
            # T[i, j, 0] = A[i, j] B[j, 0]
            rank = n * m
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i * m + j, col] = 1.0
                    V[j * p + 0, col] = 1.0
                    W[0 * n + i, col] = 1.0
                    col += 1
            return U, V, W

        # Recursive step
        best_rank = float('inf')
        best_decomp = None
        
        # Try splitting n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = get_decomposition(n1, m, p)
            U2, V2, W2 = get_decomposition(n2, m, p)
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                U = np.vstack([U1, U2])
                V = np.hstack([V1, V2])
                W = np.vstack([W1, W2])
                best_decomp = (U, V, W)
                
        # Try splitting m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = get_decomposition(n, m1, p)
            U2, V2, W2 = get_decomposition(n, m2, p)
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                # U index (i, j). j is split.
                # U1 has shape (n*m1, r1). U2 has shape (n*m2, r2).
                # We need to interleave? No, U rows are indexed by i*m+j.
                # If we split m, the rows of U for j1 and j2 are separate blocks?
                # No, for a fixed i, j runs 0..m1-1 and m1..m-1.
                # So U is not a simple vstack.
                # U[i, j] for j in m1 is in U1.
                # U[i, j] for j in m2 is in U2.
                # We need to construct U by placing U1 and U2 in the correct rows.
                U = np.zeros((n * m, best_rank))
                V = np.hstack([V1, V2])
                W = np.hstack([W1, W2]) # W index (k, i). i is not split. So W is hstack?
                # Wait, W index is (k, i). If m is split, W is unaffected in index?
                # T[i, j, k]. W[k, i, t].
                # If we split j, W is the same for both parts?
                # No, the decomposition is separate.
                # W1[k, i, t] and W2[k, i, t].
                # So W = hstack([W1, W2])?
                # Yes, because the index (k, i) is the same for both.
                # Now U.
                for i in range(n):
                    for j1 in range(m1):
                        U[i * m + j1, :r1] = U1[i * m1 + j1, :]
                    for j2 in range(m2):
                        U[i * m + m1 + j2, r1:] = U2[i * m2 + j2, :]
                best_decomp = (U, V, W)

        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = get_decomposition(n, m, p1)
            U2, V2, W2 = get_decomposition(n, m, p2)
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            total_rank = r1 + r2
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine
                U = np.hstack([U1, U2]) # U index (i, j). p is not in index. So hstack.
                V = np.hstack([V1, V2]) # V index (j, k). k is split.
                # V[j, k] for k in p1 is in V1.
                # V[j, k] for k in p2 is in V2.
                V = np.zeros((m * p, best_rank))
                for j in range(m):
                    for k1 in range(p1):
                        V[j * p + k1, :r1] = V1[j * p1 + k1, :]
                    for k2 in range(p2):
                        V[j * p + p1 + k2, r1:] = V2[j * p2 + k2, :]
                W = np.vstack([W1, W2]) # W index (k, i). k is split.
                best_decomp = (U, V, W)

        if best_decomp is None:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            best_decomp = (U, V, W)
            
        return best_decomp

    U, V, W = get_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
