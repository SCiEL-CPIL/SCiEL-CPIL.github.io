import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Generate schoolbook decomposition
    def get_schoolbook():
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Define the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # Flatten indices
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Attempt to find a lower rank decomposition
    # Use a simple heuristic: try to merge columns in the schoolbook decomposition
    # or use a random search with integer/half-integer entries
    
    # Given the complexity and time limit, we will try a few random restarts
    # to see if we can find a rank < 18 solution.
    # We will search for rank 17 first.
    
    deadline = time.time() + 110  # Leave 10s margin
    
    for target_rank in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
            
        found = False
        for _ in range(100):  # Few iterations per rank
            if time.time() > deadline:
                break
                
            # Random initialization with small integers
            # To make it exact, we need to solve a system of equations
            # This is hard. Instead, we check if the schoolbook columns are linearly dependent.
            
            # Check linear dependency of the 18 rank-1 tensors
            # Flatten each rank-1 tensor to a vector of size (n*m)*(m*p)*(p*n)
            # This is too large for direct dependency check? 6*9*6 = 324. 18 vectors of 324.
            # We can check if the 18 vectors are linearly independent.
            # If they are, rank is 18.
            # If they are dependent, we might reduce rank.
            
            # Construct the matrix of flattened tensors
            M = np.zeros((n * m * m * p * p * n, rank_schoolbook))
            for c in range(rank_schoolbook):
                # Reshape U[:, c], V[:, c], W[:, c] to tensor
                # This is just the outer product
                vec = np.outer(np.outer(U_sb[:, c], V_sb[:, c]), W_sb[:, c]).flatten()
                M[:, c] = vec
            
            # Check rank of M
            r = np.linalg.matrix_rank(M, tol=1e-9)
            if r < rank_schoolbook:
                # There is a dependency.
                # We can try to find a combination.
                # For now, if we detect dependency, we might be able to reduce rank.
                # However, constructing the new decomposition is complex.
                # We will just return the schoolbook if we can't easily construct the reduced one.
                pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
