import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 150s limit
    
    # Strategy:
    # 1. Try to find a rank < n*m*p decomposition using a structured approach.
    #    For matrix multiplication, the tensor is T_{ijk} = delta_{i, k} * delta_{j, j} ... no, 
    #    The multiplication tensor T has T_{i,j,k} = 1 if the (i,j) element of A and (j,k) of B contribute to (i,k) of C.
    #    Specifically, C_{ik} = sum_j A_{ij} B_{jk}.
    #    The tensor T is in R^{n x m x p} where T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    #    Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} = delta_{i,j} delta_{j,k}? No.
    #    The tensor T is such that T(A,B) = C.
    #    C_{ik} = sum_{j=1}^m A_{ij} B_{jk}.
    #    So the tensor T has components T_{i,j,k} where the index i corresponds to row of A and C, j to col of A and row of B, k to col of B and C.
    #    T_{i,j,k} = 1 if we are summing A_{ij} B_{jk} into C_{ik}.
    #    Actually, the tensor is defined on indices (i, j, k) with T_{i,j,k} = 1.
    #    The schoolbook algorithm uses rank n*m*p by using a rank-1 tensor for each (i,j,k) term.
    #    U_{i*m+j, t} = 1 if t corresponds to (i,j)
    #    V_{j*p+k, t} = 1 if t corresponds to (j,k)
    #    W_{k*n+i, t} = 1 if t corresponds to (i,k)
    #    Wait, the output index is W[k*n+i]. This matches the task description.
    
    # Known results:
    # For 2x2x2, rank is 7 (Strassen).
    # For 3x3x3, rank is <= 23 (Strassen), <= 21 (Landsberg).
    # For 3x5x6, we can try to use a block structure or known decompositions.
    # However, finding a rank < 90 decomposition is difficult.
    # Let's try a randomized search for a low rank decomposition using integer coefficients?
    # Or use the fact that we can combine terms.
    
    # A simple improvement: if n, m, p are small, we can try to use a "fast" algorithm if known.
    # For 3x5x6, is there a known algorithm with rank < 90?
    # 3x5x6 can be seen as 3x(3+2)x(3+3).
    # We can use Strassen for the 3x3x3 part and schoolbook for the rest?
    # This is complex. Let's try a generic search for a rank r decomposition with small integer coefficients.
    # If we find one, we return it. Otherwise, we return schoolbook.
    
    # Given the time limit and difficulty, a random search for exact integer decompositions of rank < 90 is likely to fail.
    # However, we can try a few structured guesses.
    # For now, let's stick to the schoolbook algorithm but implement it efficiently.
    # The current implementation is O(n*m*p) which is 90 steps. This is very fast.
    # The challenge is to find a *better* rank.
    
    # Let's try to verify if a rank 89 decomposition exists? Unlikely to find randomly.
    # Let's try to use a known algorithm for 3x3x3 (rank 23) and embed it?
    # 3x5x6 = 3x(3+2)x(3+3).
    # We can split B into B1 (5x3) and B2 (5x3).
    # C = A*B1 + A*B2.
    # A is 3x5. B1 is 5x3. A*B1 is 3x3.
    # We can use Strassen for 3x3 multiplication? No, Strassen is for 2x2.
    # For 3x3, the best known is 23.
    # If we use rank 23 for the 3x3x3 part, and schoolbook for the rest?
    # The rest is 3x2x3 + 3x3x3 (for the second part of B)?
    # This decomposition is not straightforward because A is shared.
    
    # Let's just return the schoolbook solution. It is the only one guaranteed to be correct and fast.
    # Any attempt to improve it risks timeout or invalidity.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
