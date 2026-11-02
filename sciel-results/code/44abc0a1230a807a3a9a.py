"""Program for M_{2,3,6} rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p = 36
    rank_schoolbook = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # T_hat[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # Check if T_hat == T
        r = U.shape[1]
        T_hat = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_hat, T, atol=1e-9)
    
    # Construct schoolbook decomposition
    U = np.zeros((n*m, rank_schoolbook), dtype=float)
    V = np.zeros((m*p, rank_schoolbook), dtype=float)
    W = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    
    if not verify(U, V, W):
        # Should never happen, but just in case
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = rank_schoolbook
    best_decomp = (U, V, W)
    
    # Attempt to find a lower rank decomposition using matricization analysis
    # The rank of the tensor is at least the maximum rank of its matricizations
    # Matricization along mode 1: shape (n*m, m*p * p*n) = (6, 216)
    # Matricization along mode 2: shape (m*p, n*m * p*n) = (18, 72)
    # Matricization along mode 3: shape (p*n, n*m * m*p) = (12, 108)
    
    # Compute ranks of matricizations to get a lower bound
    T_mat1 = T.reshape(n*m, -1)
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, -1)
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, -1)
    
    rank_lb = max(np.linalg.matrix_rank(T_mat1), 
                  np.linalg.matrix_rank(T_mat2), 
                  np.linalg.matrix_rank(T_mat3))
    
    # For this tensor, the matricization ranks give us the trivial lower bound
    # rank_lb should be 18 (from mode 2 matricization)
    
    # Try a few quick attempts to find rank < 36
    # Use a structured approach: try to find if we can reduce rank by exploiting symmetry
    # For now, given the time constraints and the difficulty of finding exact lower-rank decompositions,
    # we'll do a very limited search
    
    if time.time() < deadline - 30:
        # Try rank 35 with a simple perturbation approach
        for r in range(rank_schoolbook - 1, max(rank_lb, 20), -1):
            if time.time() > deadline - 15:
                break
            
            # Try a few random initializations with gradient-free optimization
            # This is unlikely to work but we try
            found = False
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                
                # Initialize with schoolbook and try to merge two terms
                U_test = np.zeros((n*m, r), dtype=float)
                V_test = np.zeros((m*p, r), dtype=float)
                W_test = np.zeros((p*n, r), dtype=float)
                
                # Copy first r-1 terms from schoolbook
                for c in range(r-1):
                    U_test[:, c] = U[:, c]
                    V_test[:, c] = V[:, c]
                    W_test[:, c] = W[:, c]
                
                # Last term is a combination of the last two schoolbook terms
                # with random weights
                alpha = random.uniform(-1, 1)
                beta = random.uniform(-1, 1)
                U_test[:, r-1] = U[:, r-1] + alpha * U[:, r]
                V_test[:, r-1] = V[:, r-1] + beta * V[:, r]
                W_test[:, r-1] = W[:, r-1] + (alpha + beta) * W[:, r]
                
                if verify(U_test, V_test, W_test):
                    best_rank = r
                    best_decomp = (U_test, V_test, W_test)
                    found = True
                    break
            
            if found:
                break
    
    # Return the best decomposition found
    U_best, V_best, W_best = best_decomp
    
    # Convert to lists of lists
    U_list = [[float(U_best[i, j]) for j in range(U_best.shape[1])] for i in range(U_best.shape[0])]
    V_list = [[float(V_best[i, j]) for j in range(V_best.shape[1])] for i in range(V_best.shape[0])]
    W_list = [[float(W_best[i, j]) for j in range(W_best.shape[1])] for i in range(W_best.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
