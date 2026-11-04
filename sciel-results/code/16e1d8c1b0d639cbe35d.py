"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build the matrix multiplication tensor T[i,j,k] = 1
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # The schoolbook rank is n*m*p = 48.
    # We try to find a lower rank decomposition.
    #
    # Strategy: Try rank values below 48 and use randomized search / structured ansatz.
    # For 3x4x4, the best known upper bound might be lower than 48.
    # Let's try a structured approach: use the fact that we can share factors.

    # First, let's try the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Try to find lower rank decompositions using structured search
    # Strategy: Start from schoolbook and try to merge pairs of terms
    # For exact decomposition, we need integer/half-integer entries
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Try rank reduction by 1: remove one schoolbook term and add a compensating term
    # This is equivalent to finding a rank-1 update that cancels one basis term
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(15, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        # Approach 1: Start from schoolbook and try to zero out one column
        # by adding a combination of other columns
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Create a copy of schoolbook
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            # Try to merge two terms: pick two columns and see if we can replace them
            # with a single column that preserves the tensor
            c1 = np.random.randint(0, best_rank)
            c2 = np.random.randint(0, best_rank)
            if c1 == c2:
                continue
            
            # Try to find alpha, beta, gamma such that:
            # U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] = 
            # alpha*(U[:,c1]+U[:,c2])*V[:,c1]*W[:,c1] + ... (this is getting complex)
            
            # Simpler: try random perturbations of the schoolbook
            U_new = U_sb.copy()
            V_new = V_sb.copy()
            W_new = W_sb.copy()
            
            # Add small random noise to some entries
            for _ in range(5):
                row = np.random.randint(0, U_new.shape[0])
                col = np.random.randint(0, U_new.shape[1])
                U_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                
                row = np.random.randint(0, V_new.shape[0])
                col = np.random.randint(0, V_new.shape[1])
                V_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                
                row = np.random.randint(0, W_new.shape[0])
                col = np.random.randint(0, W_new.shape[1])
                W_new[row, col] += np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            # Try to find a lower-rank approximation by SVD-like truncation
            # But we need exact decomposition, so this won't work directly
            
            # Instead, try to find if any column is redundant
            # Check if column c can be expressed as a combination of others
            for c in range(target_rank, best_rank):
                # Try to express column c as linear combination of first target_rank columns
                # This is a linear system, but we need it to work for the tensor product
                pass
        
        # Approach 2: Try known small-rank decompositions for sub-blocks
        # Embed Strassen's 2x2x2 rank-7 decomposition into the 3x4x4 problem
        if attempt == 0:  # Do this once per target_rank
            # Strassen's algorithm for 2x2 matrices:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            # C11 = M1+M4-M5+M7
            # C12 = M3+M5
            # C21 = M2+M4
            # C22 = M1-M2+M3+M6
            
            # We can use this for the 2x2 sub-blocks within our 3x4x4 problem
            # But this requires careful embedding and is complex
            
            pass
    
    # If we couldn't find better, try one more structured approach
    # Use the fact that for matrix multiplication, we can sometimes share factors
    # across different (i,j,k) tuples
    
    # Try to find a decomposition with rank 47 by solving a system of equations
    # This is a hard problem, so we'll just return the best we found
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
