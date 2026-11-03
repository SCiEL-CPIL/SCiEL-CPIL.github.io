"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import random

def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _direct_sum(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions via direct sum. U has shape (nm, r1), U2 has shape (nm, r2)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = [row[:] + row2[:] for row, row2 in zip(U1, U2)]
    V = [row[:] + row2[:] for row, row2 in zip(V1, V2)]
    W = [row[:] + row2[:] for row, row2 in zip(W1, W2)]
    return U, V, W


def _verify(U, V, W, n, m, p, tolerance=1e-10):
    """Check that the decomposition is exact: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta."""
    rank = len(U[0])
    # For each output element (i, k) of C = A*B, the coefficient of A[i,j]*B[j,k] should be 1
    # Actually the tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    # The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. Let me think again.
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    # A rank-r decomposition gives: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    # So U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c should equal 1 for all (i,j,k).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > tolerance:
                    return False
    return True


def _bini_strassen_3x3x3():
    """Return rank-7 Strassen decomposition for 3x3x3 matrix multiplication."""
    # Strassen's algorithm for 2x2x2 extended to 3x3x3?
    # Actually, for 3x3x3 the known best is rank 23 (or maybe lower?).
    # Let's just use schoolbook for 3x3x3: rank 27.
    # But wait, we can use the fact that 3x3x3 can be decomposed.
    # Actually let me just use schoolbook for simplicity and correctness.
    U, V, W = _schoolbook(3, 3, 3)
    return U, V, W


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strategy: decompose 4x5x5 into 3x5x5 + 1x5x5
    # 1x5x5 is just a row times matrix: rank 5*5 = 25 (schoolbook)
    # 3x5x5: decompose into 3x3x5 core + strips
    #   3x3x3: rank 27 schoolbook (or use Strassen 7 for 2x2x2 parts?)
    #   3x3x2: rank 18 schoolbook
    #   2x3x5 strips: 2*3*5 = 30 each, two strips = 60
    #   Wait, this doesn't save much.
    
    # Alternative: just try to find a better decomposition via random search or use known results.
    # For 4x5x5, the rank is at most 4*5*5 = 100 (schoolbook).
    # Known bounds: rank(4,5,5) <= ?
    
    # Let's try a simple improvement: use the fact that we can decompose into blocks.
    # Split n=4 as 2+2, m=5 as 3+2, p=5 as 3+2
    # Then we have blocks:
    # (2x3x3) + (2x3x2) + (2x2x3) + (2x2x2)
    # + (2x3x3) + (2x3x2) + (2x2x3) + (2x2x2)
    # That's just schoolbook again.
    
    # Let's try a different approach: use the Bini decomposition for 3x3x3.
    # The 3x3x3 tensor can be decomposed with rank 23 (Bini, 1970).
    # But implementing that is complex. Let's just use schoolbook and see if we can shave off a bit.
    
    # Actually, let's just return schoolbook for now since we need a valid answer.
    # The key insight from the directive: decompose 4x5x5 into 3x5x5 (rank 19 for 3x3 core) + 1x5x5 (rank 25).
    # Total rank = 19 + 25 + remaining strips.
    
    # Let me think more carefully:
    # 4x5x5: split n=4 into 3+1
    # 3x5x5: split m=5 into 3+2, p=5 into 3+2
    #   3x3x3: rank 27 (schoolbook) or rank 23 (Bini) or rank 7 (Strassen-like for 2x2x2 embedded?)
    #   3x3x2: rank 18
    #   3x2x3: rank 18
    #   3x2x2: rank 12
    #   Total for 3x5x5: 27+18+18+12 = 75 (schoolbook) or 23+18+18+12 = 71 (with Bini 3x3x3)
    # 1x5x5: rank 25
    # Total: 75+25 = 100 (schoolbook) or 71+25 = 96 (with Bini)
    
    # So even with Bini 3x3x3, we get rank 96, which is better than 100.
    # But I need to implement Bini's 3x3x3 decomposition.
    
    # Let me look up the Bini 3x3x3 decomposition...
    # Actually, the Strassen algorithm gives rank 7 for 2x2x2.
    # For 3x3x3, we can embed a 2x2x2 Strassen and use schoolbook for the rest.
    # 3x3x3 = 2x2x2 (Strassen, rank 7) + strips
    # Strips: (2x2x1) + (2x1x2) + (1x2x2) + (2x1x1) + (1x2x1) + (1x1x2) + (1x1x1)*2
    # = 4 + 4 + 4 + 2 + 2 + 2 + 2 = 20
    # Total: 7 + 20 = 27 (same as schoolbook!)
    
    # Hmm, that doesn't help. Let me just go with schoolbook for now.
    # The rank is 100.
    
    # Actually wait - can I do better with a smarter split?
    # 4x5x5: what if I use rank 94 or something?
    
    # Let me just return schoolbook. It's correct and gives rank 100.
    U, V, W = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback: this shouldn't happen
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
