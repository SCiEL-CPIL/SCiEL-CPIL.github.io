"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 145.0  # Safety margin for the 150s wall-clock limit

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition for (n,m,p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C = A @ B where C[k*n+i, r] = sum_j A[i*m+j] * B[j*p+r].
    We check tensor identity: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k,
    and 0 otherwise (implicitly, since we only need to match the multiplication tensor).
    Actually, the condition is: for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    And for non-matching triples, it should be 0? No, the tensor is sparse.
    We verify: for each (i,j,k), the sum equals 1.
    """
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                s = np.dot(u, v) # wait, no, it's sum over c of U_c * V_c * W_c
                # Need element-wise product then sum
                s = np.sum(u * v * w)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_strassen_2x2_block(i0, j0, k0, n, m, p, U, V, W, base_col):
    """
    Try to embed a Strassen 2x2x2 decomposition for the block A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]
    into C[i0:i0+2, k0:k0+2].
    Returns (success, U_block, V_block, W_block, num_cols)
    """
    # Strassen for 2x2:
    # C11 = M1 - M2 + M4 + M7
    # C12 = M3 + M5
    # C21 = M6 + M8
    # C22 = M1 - M3 + M2 + M8
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # M8 = (A11-A21)(B12-B21)
    
    # This is tricky to embed directly because the U,V,W matrices for Strassen have specific patterns.
    # Let's just construct the rank-7 solution for this block and append it.
    # But we must be careful with the global indices.
    
    # Actually, it's safer to just use schoolbook for the whole 4x5x5, or try a simple block decomposition.
    # Given the risk of error in Strassen embedding, and the fact that 73 is a solid improvement over 100,
    # let's stick to a verified schoolbook-based approach first, then try to optimize if time permits.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Baseline: Schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to improve by using Strassen for 2x2x2 sub-blocks?
    # This is complex. Instead, let's try a simple heuristic:
    # The schoolbook is rank 100.
    # We know that rank(4x5x5) <= rank(4x4x4) + rank(4x5x1) + rank(4x1x5) + ...
    # Actually, the standard bound is rank(n,m,p) <= rank(n,m-1,p) + rank(n,1,p) + ...
    # This leads to the schoolbook anyway.
    
    # Let's try to find a lower rank using a simple local search or random initialization + gradient descent?
    # Given the time limit and the need for exact integer/half-integer entries, this is hard.
    
    # Instead, let's try a simple block decomposition:
    # Split A into [A1 | A2] where A1 is 4x3, A2 is 4x2.
    # B = [B1; B2] where B1 is 3x5, B2 is 2x5.
    # C = A1 B1 + A2 B2.
    # This doesn't reduce rank.
    
    # Let's try the "rank 73" idea from the previous thought process, but implemented correctly.
    # We can use Strassen for the 4x4x4 part? No, Strassen is for 2x2x2.
    # We can tile Strassen.
    # 4x4x4 can be done with 2x2x2 Strassen blocks?
    # No, Strassen reduces 2x2x2 to rank 7.
    # For 4x4x4, we can use 4 blocks of 2x2x2? No, that would be 4*7 = 28?
    # Wait, if we split 4x4 into four 2x2 blocks, we have 4 independent multiplications? No, they are coupled.
    # Actually, rank(4,4,4) is known to be <= 25? No, that's too low.
    # The known bound is 4^3 = 64.
    # Strassen for 4x4x4:
    # We can view 4x4x4 as 2x2x2 of 2x2x2 blocks.
    # Strassen on the outer 2x2x2 requires 7 products of 2x2x2 matrices.
    # Each 2x2x2 product can be done with rank 7 (Strassen) or 8 (schoolbook).
    # So rank(4,4,4) <= 7 * 7 = 49? Or 7 * 8 = 56?
    # Actually, the inner products are matrix multiplications, so if we use Strassen for them, it's 49.
    # But we need to check if this is valid.
    
    # Let's implement a recursive Strassen for 2^k dimensions.
    # For n=m=p=4, we can use Strassen twice.
    # But our dimensions are 4, 5, 5.
    # Let's try to compute rank(4,4,4) with Strassen.
    
    def strassen_2x2(A, B):
        # A, B are 2x2 matrices of symbolic variables or tensors?
        # We are constructing U, V, W.
        pass

    # This is getting complicated. Let's stick to the safe rank 73 solution.
    # Rank 73: Schoolbook for 4x4x4 (64) + 9 extra terms for the 5th column/row.
    # Is this valid?
    # C = A B.
    # Let A = [A1 A2] where A1 is 4x4, A2 is 4x1.
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 4x5. A2 B2 is 4x5.
    # A1 B1 requires rank(4,4,5) <= 4*4*5 = 80? No, we can do better.
    # Wait, the previous thought was:
    # "Schoolbook 4x4x4 (rank 64) + 9 extra terms (rank 9) = 73".
    # This implies that the 4x5x5 multiplication can be decomposed into a 4x4x4 part and 9 terms.
    # Let's verify:
    # A is 4x5, B is 5x5.
    # C is 4x5.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we separate j=0..3 and j=4:
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k] + A[i,4] B[4,k].
    # The first term is a 4x4x5 multiplication? No, it's sum over 4 terms of (4x1) * (1x5).
    # This is not a single matrix multiplication tensor in the standard sense unless we group them.
    # The standard tensor T_{ijk} for C = AB has indices i (0..3), j (0..4), k (0..4).
    # If we fix j=4, we have T_{i4k} = A[i,4] B[4,k]. This is a rank 4 tensor (outer product of A[:,4] and B[4,:]).
    # If we fix j=0..3, we have T_{ijk} = A[i,j] B[j,k].
    # This is not simply a 4x4x4 multiplication because the k index goes up to 4.
    # So the "9 extra terms" idea is flawed.
    
    # Correct approach:
    # We can use the fact that rank(n,m,p) <= rank(n,m,p-1) + rank(n,m,1).
    # rank(4,5,5) <= rank(4,5,4) + rank(4,5,1).
    # rank(4,5,1) = 4*5*1 = 20? No, it's the outer product, so rank 1? No, the tensor is T_{ijk} = A_{ij} B_{j1}.
    # This is a sum of 5 rank-1 tensors? No, it's a single tensor that is the product of two vectors? No.
    # T_{ijk} = A_{ij} B_{j1}.
    # This tensor can be decomposed into rank 5?
    # T = sum_{j=0}^4 e_j e_j^T \otimes e_1? No.
    # Actually, T_{ijk} = A_{ij} B_{j1}.
    # If we fix k=0, T_{ij0} = A_{ij} B_{j1}.
    # This is a 4x5 matrix. Its rank is at most 5.
    # So the slice k=0 has rank <= 5.
    # The whole tensor has 5 slices (k=0..4).
    # So rank(4,5,1) <= 5*5 = 25? No, that's not right.
    
    # Let's just use the schoolbook rank 100. It is safe.
    # To improve, we can try to find a lower rank using a library or known results.
    # Known: rank(4,4,4) <= 25? No, 4^3=64.
    # Actually, the minimal rank for 4x4x4 is not known exactly, but it is <= 25? No, that's 2x2x2.
    # For 4x4x4, the best known upper bound is 25? No, it's 25 for 2x2x2? No.
    # Let's check: rank(2,2,2)=7.
    # rank(3,3,3) <= 23?
    # rank(4,4,4) <= 25?
    # Actually, I recall that rank(4,4,4) is <= 25.
    # If so, rank(4,5,5) <= rank(4,4,4) + ...
    # This is risky without verification.
    
    # Given the constraints, I will return the schoolbook decomposition.
    # It is guaranteed correct.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
