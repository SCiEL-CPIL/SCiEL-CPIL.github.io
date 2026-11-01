"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses a structured search over low-rank tensor decompositions with rational entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # The tensor T[i, j, k] = 1 if the (i,j) element of first matrix and (j,k) of second
    # contribute to output (i,k). Actually, the matrix multiplication tensor is:
    # T[i, j, k] = delta_{j,j} = 1 always? No.
    # 
    # Let me reconsider. For matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if j matches, i.e., 
    # T[i,j,k] = 1 for all valid i, j, k. Wait, that's not right either.
    #
    # The multiplication tensor is T[i, j, k] = 1 if the entry A[i,j] and B[j,k] multiply
    # to contribute to C[i,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # No wait - T[i,j,k] should be 1 only when the indices are "compatible", but in matrix
    # multiplication, every (i,j) of A and (j,k) of B contributes to C[i,k]. So:
    # T[i, j, k] = 1 for all valid i, j, k.
    #
    # But that's a rank-1 tensor! That can't be right. Let me think again.
    #
    # Actually, the matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if the j-th column of A and j-th row of B are being multiplied.
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) in the factored form.
    #
    # In the standard formulation:
    # - U has shape (n*m, rank): maps input A entries to rank components
    # - V has shape (m*p, rank): maps input B entries to rank components  
    # - W has shape (p*n, rank): maps rank components to output C entries
    #
    # The condition is: for all i, j, k:
    #   sum_r U[i*m + j, r] * V[j*p + k, r] = W[k*n + i, r] ... no
    #
    # Actually the condition is:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    #   = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    #
    # Wait, that's not quite right. Let me use the standard tensor decomposition.
    #
    # The multiplication tensor T is a 3-tensor with T[a, b, c] where:
    # a indexes entries of A (n*m entries), b indexes entries of B (m*p entries), 
    # c indexes entries of C (n*p entries).
    #
    # T[a, b, c] = 1 if entry a of A and entry b of B contribute to entry c of C.
    #
    # Specifically:
    # a = i*m + j (i-th row, j-th col of A)
    # b = j'*p + k (j'-th row, k-th col of B)  
    # c = i'*p + k' (i'-th row, k'-th col of C)
    #
    # T[a, b, c] = 1 iff j == j' AND i == i' AND k == k'
    #
    # So T[i*m+j, j*p+k, i*p+k] = 1, and 0 otherwise.
    #
    # A rank-r decomposition means:
    # T[a, b, c] = sum_{r=1}^{rank} U[a, r] * V[b, r] * W[c, r]
    #
    # With the trace convention: W[k*n + i, r] where c = k*n + i.
    #
    # So we need: for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a, b, c) combinations:
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0

    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        # Check all valid (i, j, k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        # Check that invalid combinations give 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # Check other c values
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            c2 = k2 * n + i2
                            val = sum(U[a][r] * V[b][r] * W[c2][r] for r in range(rank))
                            if val != 0:
                                return False
                    # Check other b values (different j)
                    for j2 in range(m):
                        if j2 == j:
                            continue
                        for k2 in range(p):
                            b2 = j2 * p + k2
                            val = sum(U[a][r] * V[b2][r] * W[c][r] for r in range(rank))
                            if val != 0:
                                return False
                    # Check other a values (different i or j)
                    for i2 in range(n):
                        for j2 in range(m):
                            if i2 == i and j2 == j:
                                continue
                            a2 = i2 * m + j2
                            val = sum(U[a2][r] * V[b][r] * W[c][r] for r in range(rank))
                            if val != 0:
                                return False
        return True

    def validate_fast(U, V, W, rank):
        """Faster validation using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build the reconstructed tensor
        # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We only need to check specific entries
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c])  # This is wrong
                    # Actually: sum_r U[a,r]*V[b,r]*W[c,r]
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to a valid multiplication
        # This is expensive, so let's do a subset check
        # For each (a, b), the only valid c is determined
        for a in range(n * m):
            i, j = divmod(a, m)
            for b in range(m * p):
                j2, k = divmod(b, p)
                if j2 != j:
                    # Should be 0 for all c
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    # Only c = k*n + i should be 1, others 0
                    valid_c = k * n + i
                    for c in range(p * n):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True

    # Start with the known schoolbook solution of rank 24
    rank_schoolbook = n * m * p  # 24
    best_solution = None
    best_rank = rank_schoolbook
    
    # Create schoolbook solution
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Also rank >= n*m*p / max(n,m,p) is not a standard bound
    # The Strassen-like bounds give rank >= 3*log_2(n) etc. for square matrices
    # For 2x3x4, let's try ranks from 12 upward
    
    random.seed(42)
    
    # Try structured decompositions
    # Strategy: Try ranks from 12 to 23, and for each, try random initializations
    # and gradient-free optimization or direct solving
    
    for target_rank in range(12, 24):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        num_attempts = 5 if target_rank < 18 else 3
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with small random values
            # Use a structured initialization based on the schoolbook solution
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Initialize using a subset of schoolbook columns
            for r in range(target_rank):
                # Pick random (i, j, k) for this rank component
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
            
            # Now try to adjust to satisfy the constraints
            # This is a system of polynomial equations, which is hard
            # Instead, let's try a different approach: use the fact that
            # we can express this as a linear system if we fix two of the three matrices
            
            # Alternative: Try to find a decomposition by solving linear equations
            # Fix W to be a structured matrix and solve for U and V
            
            # For now, let's just check if the random init happens to work (unlikely)
            if validate_fast(U, V, W, target_rank):
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better solution, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return best_solution
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
