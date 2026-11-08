import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = 3, 3, 4
    
    # Build the target tensor
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(4):
                T_expected[i*3+j, j*4+k, k*3+i] = 1
                
    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)

    # Start with the known rank-34 construction
    rank = 34
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    col = 0
    # Strassen Left (A11 * B1L)
    U[0, col] = 1; U[4, col] = 1; V[0, col] = 1; V[5, col] = 1; W[0, col] = 1; W[4, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[0, col] = 1; W[1, col] = 1; W[4, col] = -1; col += 1
    U[0, col] = 1; V[1, col] = 1; V[5, col] = -1; W[3, col] = 1; W[4, col] = 1; col += 1
    U[4, col] = 1; V[4, col] = 1; V[0, col] = -1; W[0, col] = 1; W[1, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[5, col] = 1; W[0, col] = -1; W[3, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[0, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[4, col] = 1; V[5, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    # Strassen Right (A11 * B1R)
    U[0, col] = 1; U[4, col] = 1; V[2, col] = 1; V[7, col] = 1; W[6, col] = 1; W[10, col] = 1; col += 1
    U[3, col] = 1; U[4, col] = 1; V[2, col] = 1; W[7, col] = 1; W[10, col] = -1; col += 1
    U[0, col] = 1; V[3, col] = 1; V[7, col] = -1; W[9, col] = 1; W[10, col] = 1; col += 1
    U[4, col] = 1; V[6, col] = 1; V[2, col] = -1; W[6, col] = 1; W[7, col] = 1; col += 1
    U[0, col] = 1; U[1, col] = 1; V[7, col] = 1; W[6, col] = -1; W[9, col] = 1; col += 1
    U[3, col] = 1; U[0, col] = -1; V[2, col] = 1; V[3, col] = 1; W[10, col] = 1; col += 1
    U[1, col] = 1; U[4, col] = -1; V[6, col] = 1; V[7, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1

    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 34
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank by checking for linear dependencies among the rank-1 terms
    # Vectorize the outer products and check column rank
    def get_vectorized_terms(U, V, W):
        R = U.shape[1]
        terms = np.zeros((9 * 12 * 12, R), dtype=int)
        for r in range(R):
            term = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            terms[:, r] = term.flatten()
        return terms

    terms = get_vectorized_terms(U, V, W)
    current_rank = U.shape[1]
    
    # Iteratively remove dependent columns
    # We check if the matrix of terms has full column rank.
    # If not, we find a column that is a linear combination of others and remove it.
    # Since the entries are integers, we use QR or SVD with tolerance.
    
    for _ in range(10): # Max 10 reductions
        if time.time() > deadline - 10:
            break
        if current_rank <= 1:
            break
            
        # Calculate rank of the term matrix
        # Use SVD for numerical stability
        try:
            U_svd, s, Vt = np.linalg.svd(terms, full_matrices=False)
            tol = 1e-9
            rank_svd = np.sum(s > tol)
            
            if rank_svd < current_rank:
                # Find a dependent column
                # The columns corresponding to small singular values are dependent.
                # We can identify the column with the largest coefficient in the null space?
                # Simpler: Try removing each column and check if the rank stays the same.
                removed = False
                for r in range(current_rank):
                    if time.time() > deadline - 10:
                        break
                    mask = [i for i in range(current_rank) if i != r]
                    terms_test = terms[:, mask]
                    # Check rank of reduced set
                    if terms_test.shape[1] == 0:
                        break
                    U_test, s_test, Vt_test = np.linalg.svd(terms_test, full_matrices=False)
                    rank_test = np.sum(s_test > tol)
                    
                    if rank_test == rank_svd:
                        # Removing column r did not reduce the rank of the term set
                        # This means column r was dependent on the others.
                        # We can remove it from U, V, W and the term matrix.
                        U = U[:, mask]
                        V = V[:, mask]
                        W = W[:, mask]
                        terms = terms_test
                        current_rank -= 1
                        removed = True
                        break
                if not removed:
                    break
            else:
                break
        except Exception:
            break

    # Verify the reduced solution
    if current_rank < best_rank:
        if verify(U, V, W):
            best_rank = current_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if best_rank < 34:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        rank = 34

    if not verify(U, V, W):
        # Fallback
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
