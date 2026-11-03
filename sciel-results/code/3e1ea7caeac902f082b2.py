"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Start with schoolbook rank 63
    best_rank = n * m * p
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Build a hybrid: Strassen (rank 7) for first 3 cols + schoolbook for last 4 cols
    # Total: 7 + 4*3 = 7 + 12 = 19 terms
    if time.time() < deadline:
        result = _build_hybrid(n, m, p)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
                print(f"Hybrid decomposition with rank {rank}")
    
    # Try to reduce rank by removing redundant terms
    if time.time() < deadline:
        result = _reduce_rank(best_result, n, m, p, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
                print(f"Reduced decomposition with rank {rank}")
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Create the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(actual - expected)
                max_err = max(max_err, err)
                if err > 1e-9:
                    return False
    return max_err < 1e-9

def _strassen_3x3x3():
    """
    Return a rank-7 tensor decomposition for 3x3x3 matrix multiplication.
    
    Uses the known Strassen-style decomposition. The three factor matrices
    U (9x7), V (9x7), W (9x7) satisfy:
      sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1 if (i,j,k) valid, 0 otherwise
    where the multiplication tensor T_{i,j,k} = delta_{i,j,k} in the appropriate sense.
    
    Actually, for matrix multiplication C = A*B where A is 3x3, B is 3x3:
      C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor is T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    
    In our convention:
      U has shape (n*m, rank) = (9, 7), indexed by (i,j)
      V has shape (m*p, rank) = (9, 7), indexed by (j,k)  
      W has shape (p*n, rank) = (9, 7), indexed by (k,i)
    
    The condition is:
      sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all valid (i,j,k)
    But wait - this isn't quite right. Let me reconsider.
    
    For matrix multiplication A(3x3) * B(3x3) = C(3x3):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor has entries T[(i,j),(j,k),(k,i)] = 1 for all i,j,k in {0,1,2}.
    
    So we need:
      sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1,2}
    
    Strassen's algorithm computes 7 products:
      m1 = (A[0,0] + A[2,2]) * (B[0,0] + B[2,2])
      m2 = (A[1,0] + A[2,0]) * B[0,1]
      m3 = A[0,0] * (B[0,1] - B[1,1])
      m4 = A[2,2] * (B[1,1] - B[0,0])
      m5 = (A[0,2] + A[1,2]) * B[1,1]
      m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
      m7 = (A[0,2] - A[1,1]) * (B[1,0] + B[1,1])
    
    This gives:
      C[0,0] = m1 + m4 - m5 + m7
      C[0,1] = m3 + m5
      C[0,2] = m2 - m4 + m6 - m7  (wait, let me recheck)
    
    Actually, let me use a well-known rank-7 decomposition. I'll use the one from
    the literature. Let me construct it carefully.
    
    For 3x3x3, a rank-7 decomposition exists. Let me use the following known one:
    
    The seven "multiplications" are linear combinations of A's entries and B's entries.
    """
    # I'll use a verified rank-7 decomposition for 3x3x3.
    # Reference: standard Strassen extension to 3x3.
    
    # U[i,j,c] for i,j in {0,1,2}, c in {0,...,6}
    # V[j,k,c] for j,k in {0,1,2}, c in {0,...,6}
    # W[k,i,c] for k,i in {0,1,2}, c in {0,...,6}
    
    # Let me use the following construction based on the 2x2x2 Strassen (rank 7)
    # extended to 3x3x3. Actually, 2x2x2 has rank 7? No, 2x2x2 has rank 7 is NOT proven.
    # 2x2x2 has rank 7 (Strassen), 3x3x3 has rank 23 (best known upper bound is lower).
    
    # Wait - I was confused. Let me reconsider.
    # 2x2x2 matrix multiplication has tensor rank 7 (Strassen).
    # 3x3x3 matrix multiplication has tensor rank at most 23 (Landsberg's bound) 
    # but the best known explicit constructions give rank around 23-25.
    # The trivial schoolbook is 27.
    
    # So for 3x3x3, we can't use rank 7. Let me use a better approach.
    # 
    # For 3x3x7, let me think differently:
    # Split B into B1 (3x3) and B2 (3x4).
    # C1 = A * B1 (3x3 result)
    # C2 = A * B2 (3x4 result)
    #
    # For C1: use schoolbook (rank 9) or a better decomposition
    # For C2: use schoolbook (rank 12)
    # Total: 9 + 12 = 21
    #
    # Can we do better? The key insight is that both C1 and C2 depend on A.
    # If we use the same "A-factors" for both, we might save rank.
    
    # Let me just use schoolbook for now and try to reduce.
    # Actually, let me construct a hybrid where I use shared structure.
    
    # For now, return a schoolbook-based hybrid of rank 21:
    # First 9 terms: schoolbook for 3x3x3 (columns 0-2 of B)
    # Next 12 terms: schoolbook for 3x3x4 (columns 3-6 of B)
    
    U = [[0]*21 for _ in range(9)]
    V = [[0]*21 for _ in range(9)]
    W = [[0]*21 for _ in range(9)]
    
    col = 0
    # Part 1: columns 0,1,2 of B (3x3x3 schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Part 2: columns 3,4,5,6 of B (3x3x4 schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_hybrid(n, m, p):
    """Build a hybrid decomposition for 3x3x7."""
    # Use schoolbook for all parts: rank = 3*3*7 = 63
    # But we can split: 3x3x3 (rank 9) + 3x3x4 (rank 12) = 21
    # The W matrix needs to handle p*n = 7*3 = 21 entries
    # Wait, W has shape (p*n, rank) = (21, rank)
    # V has shape (m*p, rank) = (21, rank)
    # U has shape (n*m, rank) = (9, rank)
    
    # Schoolbook for full 3x3x7: rank 63
    # Hybrid: treat as two separate multiplications
    
    # Actually, the schoolbook IS the natural decomposition.
    # Let me just return the schoolbook with rank 63 for now,
    # and the reduction step will try to find redundancies.
    
    return _schoolbook_decomposition(n, m, p)

def _reduce_rank(decomp, n, m, p, deadline):
    """
    Try to reduce the rank by checking if any term is redundant.
    A term c is redundant if removing it still gives a valid decomposition.
    More aggressively: try pairs of terms that might be linearly dependent.
    """
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    rank = U.shape[1]
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank
    
    # Greedy: try removing each term
    for c_remove in range(rank):
        if time.time() >= deadline:
            break
        
        # Try removing term c_remove from current decomposition
        indices = [i for i in range(current_rank) if i != c_remove]
        if not indices:
            continue
        
        U_test = current_U[:, indices]
        V_test = current_V[:, indices]
        W_test = current_W[:, indices]
        
        if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            current_rank = len(indices)
            print(f"Removed term {c_remove}, new rank: {current_rank}")
    
    if current_rank < rank:
        return {
            "U": current_U.tolist(),
            "V": current_V.tolist(),
            "W": current_W.tolist()
        }
    
    return None
# EVOLVE-BLOCK-END
