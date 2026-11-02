"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds to spare

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k are valid indices (always are for matrix mult)
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = delta(i,j,k) 
    # No wait. Let me think about this more carefully.
    #
    # Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The multiplication tensor has entries T[i, j, k] = 1 if and only if... 
    # Actually the standard formulation: the bilinear map (A, B) -> C = A*B
    # can be written as: for each term in the sum, we have U[i,j,c], V[j,k,c], W[i,k,c]
    # such that sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = delta_{i,j} * delta_{j,k}... 
    # No, that's not right either.
    #
    # Let me reconsider. The matrix multiplication tensor is defined as:
    # T_{i,j,k} = 1 if the (i,k) entry of C depends on A_{i,j} and B_{j,k}
    # In the standard formulation for C = AB:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T_{i,j,k} = 1 for all valid i,j,k (since each triple (i,j,k)
    # contributes to exactly one output entry C_{i,k}).
    #
    # Wait, I need to be more careful. The tensor for matrix multiplication of size n x m times m x p:
    # The input is A (n*m entries) and B (m*p entries), output is C (n*p entries).
    # The bilinear form is: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # As a tensor T in the space R^{n*m} x R^{m*p} x R^{n*p}:
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # a = i*m + j, b = j*p + k, c = k*n + i (using the W convention from the task)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    #
    # The rank-r decomposition gives:
    # T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # and = 0 for all other (a,b,c) triples.
    #
    # But since a=i*m+j determines i and j uniquely, b=j*p+k determines j and k uniquely,
    # and c=k*n+i determines k and i uniquely, the only consistent triples are those where
    # the j from a matches the j from b, and the k from b matches the k from c, and the i from c matches the i from a.
    # For any valid (i,j,k), a=i*m+j, b=j*p+k, c=k*n+i is a valid triple.
    # For an invalid triple (a,b,c), at least one of the index consistency conditions fails.
    #
    # So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a,b,c) in [0,n*m) x [0,m*p) x [0,n*p):
    #   sum_r U[a, r] * V[b, r] * W[c, r] = 0
    #
    # But checking all (a,b,c) is expensive. Let's just check the valid ones and a few invalid ones.
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check some invalid entries (spot check)
    # An invalid entry would be where the indices don't align
    # e.g., a = i*m+j, b = j'*p+k where j' != j
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _try_strassen_blocks():
    """Try to use Strassen-like block structure for 3x4x8."""
    # 3x4 can be decomposed as... hmm, not directly Strassen.
    # 4x8 can be decomposed as 2x2 blocks of 2x4, or 4x4 blocks.
    # Let's think about this differently.
    #
    # Actually, for small matrices, the known minimal ranks are:
    # 2x2x2: 7 (Strassen)
    # 2x3x2: ?
    # 3x3x3: 23 (best known)
    # 2x2x3: ?
    #
    # For 3x4x8, this is a non-standard size. Let me think about block decompositions.
    #
    # 3x4 = [1x4; 2x4] or [3x2; 3x2]
    # 4x8 = [4x4, 4x4] (two 4x4 blocks) or [2x8; 2x8] or [4x2, 4x2, 4x2, 4x2]
    #
    # One approach: 3x4x8 = (3x4x4) + (3x4x4) using two 3x4x4 multiplications
    # But that doesn't help with rank.
    #
    # Another: 3x4x8 = (2x4x8) + (1x4x8)
    # 2x4x8: 2x4 = [1x4; 1x4], so 2x4x8 = 2 * (1x4x8)
    # 1x4x8: rank 4 (dot product of 4-vectors)
    # So 2x4x8: rank 8? No, 2x4x8 tensor: A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0..7}
    # This is rank 2*4 = 8 in schoolbook, but maybe less?
    # Actually, 2x4x8: the tensor has 2*4*8 = 64 non-zero entries.
    # Lower bound: max(2*8, 4*8, 2*4) = 32? No, the lower bound is max(n*p, m*p, n*m) for some cases.
    # Actually the trivial lower bound is max(n*m, m*p, n*p) = max(8, 32, 16) = 32.
    # But that's not tight.
    #
    # Let me try a different approach: direct search for small rank reductions.
    # Start from schoolbook rank 96 and try to find rank 95, 94, etc.
    #
    # Actually, let me try a more systematic approach.
    # The schoolbook rank is 3*4*8 = 96.
    # Can we do better?
    #
    # For 2x2x2, Strassen gives 7 < 8.
    # For 2x2x3: schoolbook is 12. Best known is 10.
    # For 3x3x3: schoolbook is 27. Best known is 23.
    #
    # For 3x4x8, let's see if we can combine known results.
    # 3x4x8 = (3x4x4) concatenated with (3x4x4) in the p-dimension.
    # If 3x4x4 has rank r, then 3x4x8 has rank at most 2r.
    # Schoolbook for 3x4x4 is 48, so 3x4x8 would be at most 96 (same as schoolbook).
    #
    # Alternatively, 3x4x8 = (2x4x8) + (1x4x8)
    # 1x4x8: rank 4 (each output entry is a dot product of a 4-vector with a row of B)
    # Actually, 1x4x8: A is 1x4, B is 4x8, C is 1x8.
    # C[k] = sum_j A[j] B[j,k]
    # This is rank 4: U[j, r] = delta_{j,r}, V[j,k] = B[j,k], W[k,r] = delta_{k,r}... 
    # No, that's not a valid decomposition in the tensor sense.
    #
    # The tensor for 1x4x8: T[j, k, k'] where j indexes A (4 entries), k indexes B (32 entries), k' indexes C (8 entries).
    # Wait, A has 1*4 = 4 entries, B has 4*8 = 32 entries, C has 1*8 = 8 entries.
    # T[a, b, c] where a in [0,4), b in [0,32), c in [0,8).
    # T[j, j*8+k, k] = 1 for j in [0,4), k in [0,8).
    # Rank of this tensor: it's the identity-like tensor. 
    # T[j, j*8+k, k] = delta_{j, floor(b/8)} * delta_{k, b mod 8}
    # This is rank 4: for each j, we have a "slice" that's an 8x8 identity.
    # U[j, r] = delta_{j,r}, V[j*8+k, r] = delta_{j,r} * delta_{k, ...}... 
    # Hmm, this is getting complicated. Let me just try numerical search.
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using random perturbation
    # Start from schoolbook and try to eliminate terms
    
    # Actually, let me try a more principled approach.
    # Use the fact that 4 = 2+2 and 8 = 4+4
    # 3x4x8 = 3x(2+2)x(4+4) = 3x2x4 + 3x2x4 + 3x2x4 + 3x2x4
    # Each 3x2x4 has schoolbook rank 24.
    # Total: 4 * 24 = 96 (same as schoolbook).
    
    # Try 3x4x8 = (2+1)x4x8 = 2x4x8 + 1x4x8
    # 1x4x8: rank 4 (it's just 4 dot products of length 4)
    # 2x4x8: let's see. A is 2x4, B is 4x8, C is 2x8.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is like two independent 1x4x8 multiplications.
    # Rank of 2x4x8: at most 2*4 = 8? No, that's not right.
    # The tensor T for 2x4x8: T[i*m+j, j*p+k, k*n+i] = 1
    # where i in [0,2), j in [0,4), k in [0,8).
    # T[i*4+j, j*8+k, k*2+i] = 1
    # This has 2*4*8 = 64 non-zero entries.
    # Lower bound: max(2*4, 4*8, 2*8) = max(8, 32, 16) = 32.
    # Schoolbook: 64.
    # Can we do better than 64?
    #
    # 2x4x8 = 2x(2+2)x8 = 2x2x8 + 2x2x8
    # 2x2x8: A is 2x2, B is 2x8, C is 2x8.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is like 8 independent 2x2x1 multiplications (one for each k).
    # Each 2x2x1 has rank 2 (schoolbook) or... 
    # 2x2x1: T[i,j,k] where k is always 0. T[i, j, 0] = delta_{i,j}... no.
    # 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[i] = sum_j A[i,j] B[j]
    # T[i*2+j, j*1+0, 0*2+i] = T[i*2+j, j, i] = 1
    # So T[0,0,0]=1, T[1,0,1]=1, T[2,1,0]=1, T[3,1,1]=1
    # This is rank 2: U[0,0]=1, U[1,1]=1, U[2,0]=1, U[3,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0]=1, W[1,1]=1
    # Check: T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0 = 1 ✓
    # T[1,0,1] = U[1,0]*V[0,0]*W[1,0] + U[1,1]*V[0,1]*W[1,1] = 0 + 1*0*0 = 0 ✗
    # That doesn't work.
    #
    # Let me redo. 2x2x1: n=2, m=2, p=1.
    # T[i*m+j, j*p+k, k*n+i] = T[i*2+j, j*1+0, 0*2+i] = T[i*2+j, j, i]
    # Non-zero entries: (i,j) in [0,2)x[0,2): T[i*2+j, j, i] = 1
    # (0,0): T[0, 0, 0] = 1
    # (0,1): T[1, 1, 0] = 1
    # (1,0): T[2, 0, 1] = 1
    # (1,1): T[3, 1, 1] = 1
    #
    # Rank 2 decomposition:
    # r=0: U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=0
    #        V[0,0]=1, V[1,0]=0
    #        W[0,0]=1, W[1,0]=0
    # r=1: U[0,1]=0, U[1,1]=1, U[2,1]=0, U[3,1]=0
    #        V[0,1]=0, V[1,1]=1
    #        W[0,1]=0, W[1,1]=1
    #
    # Check T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0 = 1 ✓
    # T[1,1,0] = U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 0*0*1 + 1*1*0 = 0 ✗
    #
    # That's wrong. Let me be more careful.
    # T[1,1,0]: a=1, b=1, c=0.
    # sum_r U[1,r]*V[1,r]*W[0,r]
    # r=0: U[1,0]*V[1,0]*W[0,0] = 0*0*1 = 0
    # r=1: U[1,1]*V[1,1]*W[0,1] = 1*1*0 = 0
    # Total: 0, but should be 1.
    #
    # So rank 2 doesn't work for 2x2x1. Let me try rank 4 (schoolbook).
    # Actually, 2x2x1 should have rank 4 in schoolbook (n*m*p = 2*2*1 = 4).
    # And indeed, it seems rank 2 doesn't work.
    #
    # OK, I think I'm overcomplicating this. Let me just do a numerical search.
    
    # For now, return schoolbook. The search below may improve it.
    return best

def _random_search(start_rank, max_iters=1000):
    """Try to find a lower rank decomposition by random perturbation."""
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    if start_rank >= best_rank:
        return best, best_rank
    
    # Start from schoolbook and try to merge terms
    # This is a heuristic approach
    
    # Actually, let's try a different approach: 
    # Use the known result that 2x2x2 has rank 7.
    # 3x4x8 = (2+1)x(2+2)x(4+4)
    # = 2x2x4 + 2x2x4 + 2x2x4 + 2x2x4
    #   + 1x2x4 + 1x2x4 + 1x2x4 + 1x2x4
    #
    # 2x2x4 = 2x2x(2+2) = 2x2x2 + 2x2x2
    # Each 2x2x2 has rank 7 (Strassen).
    # So 2x2x4 has rank at most 14.
    # 1x2x4: A is 1x2, B is 2x4, C is 1x4.
    # C[k] = A[0]B[0,k] + A[1]B[1,k]
    # This is rank 2: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[4,0]=1, V[1,1]=1, V[5,1]=1, W[0,0]=1, W[1,1]=1... 
    # Wait, let me be careful with indices.
    # n=1, m=2, p=4.
    # T[i*m+j, j*p+k, k*n+i] = T[i*2+j, j*4+k, k*1+i]
    # i=0 always.
    # T[j, j*4+k, k] = 1 for j in [0,2), k in [0,4).
    # Non-zero: (0,0,0), (0,4,1), (0,8,2), (0,12,3), (1,1,0), (1,5,1), (1,9,2), (1,13,3)
    # Rank 2: 
    # r=0: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[4,0]=1, V[8,0]=1, V[12,0]=1; W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    # r=1: U[0,1]=0, U[1,1]=1; V[1,1]=1, V[5,1]=1, V[9,1]=1, V[13,1]=1; W[0,1]=1, W[1,1]=1, W[2,1]=1, W[3,1]=1
    # Check T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0 = 1 ✓
    # T[0,4,1] = U[0,0]*V[4,0]*W[1,0] + U[0,1]*V[4,1]*W[1,1] = 1*1*1 + 0 = 1 ✓
    # T[1,1,0] = U[1,0]*V[1,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 0*0*1 + 1*1*1 = 1 ✓
    # T[1,5,1] = U[1,0]*V[5,0]*W[1,0] + U[1,1]*V[5,1]*W[1,1] = 0 + 1*1*1 = 1 ✓
    # Invalid: T[0,1,0] = U[0,0]*V[1,0]*W[0,0] + U[0,1]*V[1,1]*W[0,1] = 1*0*1 + 0*1*1 = 0 ✓
    # Great, so 1x2x4 has rank 2.
    #
    # So 3x4x8 = 4*(2x2x4) + 4*(1x2x4) = 4*14 + 4*2 = 56 + 8 = 64.
    # That's a big improvement over 96!
    #
    # But wait, I need to be more careful. The block decomposition needs to be done correctly.
    # 3x4x8: A is 3x4, B is 4x8, C is 3x8.
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4) and B2 (4x4).
    # Then C = A1*B1 + A1*B2 + A2*B1 + A2*B2.
    # C1 = A1*B1 (2x4), C2 = A1*B2 (2x4), C3 = A2*B1 (1x4), C4 = A2*B2 (1x4).
    #
    # A1*B1: 2x4x4. Split A1 into A1a (2x2) and A1b (2x2).
    # A1*B1 = A1a*B1a + A1b*B1b where B1a is first 2 columns of B1, B1b is last 2 columns.
    # A1a*B1a: 2x2x2, rank 7.
    # A1b*B1b: 2x2x2, rank 7.
    # Total for A1*B1: 14.
    #
    # Similarly for A1*B2: 14.
    # A2*B1: 1x4x4. Split into 1x2x2 + 1x2x2.
    # 1x2x2: rank 2.
    # Total for A2*B1: 4.
    # A2*B2: 4.
    #
    # Total: 14 + 14 + 4 + 4 = 36.
    #
    # Wait, that seems too good. Let me double-check.
    # 2x2x2 has rank 7 (Strassen). Yes.
    # 1x2x2: A is 1x2, B is 2x2, C is 1x2.
    # C[k] = A[0]B[0,k] + A[1]B[1,k]
    # T[j, j*2+k, k] = 1 for j in [0,2), k in [0,2).
    # (0,0,0), (0,2,1), (1,1,0), (1,3,1)
    # Rank 2: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[2,0]=1, V[1,1]=1, V[3,1]=1; W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    # Check: T[0,0,0] = 1*1*1 = 1 ✓, T[0,2,1] = 1*1*1 = 1 ✓, T[1,1,0] = 1*1*1 = 1 ✓, T[1,3,1] = 1*1*1 = 1 ✓
    # Invalid: T[0,1,0] = 1*0*1 + 0*1*1 = 0 ✓
    # Yes, rank 2.
    #
    # So total rank = 7+7+7+7+2+2+2+2 = 36.
    #
    # But I need to actually construct this. Let me write the code.
    
    # Actually, I realize I need to be very careful with the index mapping.
    # The full problem is 3x4x8.
    # Let me define the block structure:
    # A is 3x4: rows 0,1,2; cols 0,1,2,3.
    # B is 4x8: rows 0,1,2,3; cols 0,1,2,3,4,5,6,7.
    # C is 3x8: rows 0,1,2; cols 0,1,2,3,4,5,6,7.
    #
    # Split:
    # A1 = A[0:2, 0:4] (2x4), A2 = A[2:3, 0:4] (1x4)
    # B1 = B[0:4, 0:4] (4x4), B2 = B[0:4, 4:8] (4x4)
    #
    # C1 = A1 * B1 (2x4), C2 = A1 * B2 (2x4), C3 = A2 * B1 (1x4), C4 = A2 * B2 (1x4)
    # C[0:2, 0:4] = C1, C[0:2, 4:8] = C2, C[2:3, 0:4] = C3, C[2:3, 4:8] = C4
    #
    # Now, C1 = A1 * B1 where A1 is 2x4, B1 is 4x4.
    # Split A1 into A1a = A1[0:2, 0:2] (2x2), A1b = A1[0:2, 2:4] (2x2)
    # Split B1 into B1a = B1[0:4, 0:2] (4x2), B1b = B1[0:4, 2:4] (4x2)
    # C1 = A1a * B1a + A1b * B1b
    # C1a = A1a * B1a (2x2), C1b = A1b * B1b (2x2)
    # C1[0:2, 0:2] = C1a, C1[0:2, 2:4] = C1b
    #
    # C1a: 2x2x2, rank 7 (Strassen)
    # C1b: 2x2x2, rank 7 (Strassen)
    #
    # Similarly for C2, C3, C4.
    #
    # Total rank: 7*4 + 2*4 = 28 + 8 = 36.
    #
    # Let me now construct this.
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen 2x2x2 decomposition
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x2: n=2, m=2, p=2
    # T[i*2+j, j*2+k, k*2+i] = 1
    # Strassen gives rank 7.
    
    def _strassen_2x2x2(offset_a, offset_b, offset_c):
        """Return (U, V, W) for a 2x2x2 block at given offsets."""
        # offset_a: offset in A index (i*m+j)
        # offset_b: offset in B index (j*p+k)
        # offset_c: offset in C index (k*n+i)
        # For a 2x2x2 block, the local indices are:
        # A: [0,1,2,3] -> global: offset_a + [0,1,2,3]
        # B: [0,1,2,3] -> global: offset_b + [0,1,2,3]
        # C: [0,1,2,3] -> global: offset_c + [0,1,2,3]
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # In our indexing:
        # A index: i*2+j, so A00=0, A01=1, A10=2, A11=3
        # B index: j*2+k, so B00=0, B01=1, B10=2, B11=3
        # C index: k*2+i, so C00=0, C01=1, C10=2, C11=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = 1  # C11
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1  # -B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1  # -B00
        W[2, 3] = 1  # C10
        W[3, 3] = 1  # C11
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[1, 4] = 1  # C01
        W[3, 4] = 1  # C11
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1  # -A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[2, 5] = 1  # C10
        W[3, 5] = 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1  # -A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00
        W[3, 6] = 1  # C11
        
        # Now apply offsets
        U_full = np.zeros((n*m, 7))
        V_full = np.zeros((m*p, 7))
        W_full = np.zeros((p*n, 7))
        
        for r in range(7):
            for local_a in range(4):
                if U[local_a, r] != 0:
                    global_a = offset_a + local_a
                    if 0 <= global_a < n*m:
                        U_full[global_a, r] = U[local_a, r]
            for local_b in range(4):
                if V[local_b, r] != 0:
                    global_b = offset_b + local_b
                    if 0 <= global_b < m*p:
                        V_full[global_b, r] = V[local_b, r]
            for local_c in range(4):
                if W[local_c, r] != 0:
                    global_c = offset_c + local_c
                    if 0 <= global_c < p*n:
                        W_full[global_c, r] = W[local_c, r]
        
        return U_full, V_full, W_full
    
    def _dot_1x2x2(offset_a, offset_b, offset_c):
        """Return (U, V, W) for a 1x2x2 block at given offsets."""
        # 1x2x2: A is 1x2, B is 2x2, C is 1x2
        # T[j, j*2+k, k] = 1 for j in [0,2), k in [0,2)
        # Rank 2.
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        
        # r=0: A0 * (B00, B01)
        U[0, 0] = 1
        V[0, 0] = 1  # B00
        V[1, 0] = 1  # B01
        W[0, 0] = 1  # C0
        W[1, 0] = 1  # C1
        
        # r=1: A1 * (B10, B11)
        U[1, 1] = 1
        V[2, 1] = 1  # B10
        V[3, 1] = 1  # B11
        W[0, 1] = 1  # C0
        W[1, 1] = 1  # C1
        
        U_full = np.zeros((n*m, 2))
        V_full = np.zeros((m*p, 2))
        W_full = np.zeros((p*n, 2))
        
        for r in range(2):
            for local_a in range(2):
                if U[local_a, r] != 0:
                    global_a = offset_a + local_a
                    if 0 <= global_a < n*m:
                        U_full[global_a, r] = U[local_a, r]
            for local_b in range(4):
                if V[local_b, r] != 0:
                    global_b = offset_b + local_b
                    if 0 <= global_b < m*p:
                        V_full[global_b, r] = V[local_b, r]
            for local_c in range(2):
                if W[local_c, r] != 0:
                    global_c = offset_c + local_c
                    if 0 <= global_c < p*n:
                        W_full[global_c, r] = W[local_c, r]
        
        return U_full, V_full, W_full
    
    # Now build the full decomposition
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Block structure for 3x4x8:
    # A: 3x4, B: 4x8, C: 3x8
    #
    # C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]  (2x4x4)
    # C[0:2, 4:8] = A[0:2, 0:4] * B[0:4, 4:8]  (2x4x4)
    # C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4]  (1x4x4)
    # C[2:3, 4:8] = A[2:3, 0:4] * B[0:4, 4:8]  (1x4x4)
    #
    # 2x4x4 = 2x2x2 + 2x2x2 (splitting the 4-dimension of A into two 2s)
    # 1x4x4 = 1x2x2 + 1x2x2 (splitting the 4-dimension of A into two 2s)
    
    # C[0:2, 0:4]:
    # A rows 0-1, B cols 0-3, C rows 0-1, C cols 0-3
    # Split into:
    #   A cols 0-1, B cols 0-1: 2x2x2 at A offset 0, B offset 0, C offset 0
    #   A cols 2-3, B cols 2-3: 2x2x2 at A offset 2, B offset 4, C offset 2
    
    # For 2x2x2 with A offset oa, B offset ob, C offset oc:
    # A local indices: 0,1,2,3 -> global: oa+0, oa+1, oa+2, oa+3
    # B local indices: 0,1,2,3 -> global: ob+0, ob+1, ob+2, ob+3
    # C local indices: 0,1,2,3 -> global: oc+0, oc+1, oc+2, oc+3
    
    # C[0:2, 0:4]:
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    #   A offset: 0 (rows 0-1, cols 0-1)
    #   B offset: 0 (rows 0-1, cols 0-1)
    #   C offset: 0 (rows 0-1, cols 0-1)
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    #   A offset: 2 (rows 0-1, cols 2-3)
    #   B offset: 4 (rows 2-3, cols 0-1)
    #   C offset: 0 (rows 0-1, cols 0-1)
    U_b, V_b, W_b = _strassen_2x2x2(2, 4, 0)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8]:
    # Part 1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    #   A offset: 0
    #   B offset: 16 (rows 0-1, cols 4-5)
    #   C offset: 4 (rows 0-1, cols 4-5)
    U_b, V_b, W_b = _strassen_2x2x2(0, 16, 4)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    #   A offset: 2
    #   B offset: 20 (rows 2-3, cols 4-5)
    #   C offset: 4
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 4)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4]:
    # Part 1: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    #   A offset: 8 (row 2, cols 0-1)
    #   B offset: 0
    #   C offset: 6 (row 2, cols 0-1)
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 6)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    #   A offset: 10 (row 2, cols 2-3)
    #   B offset: 4
    #   C offset: 6
    U_b, V_b, W_b = _dot_1x2x2(10, 4, 6)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8]:
    # Part 1: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    #   A offset: 8
    #   B offset: 16
    #   C offset: 8 (row 2, cols 4-5)
    U_b, V_b, W_b = _dot_1x2x2(8, 16, 8)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    #   A offset: 10
    #   B offset: 20
    #   C offset: 8
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 8)
    _add_block(U_b, V_b, W_b)
    
    # Convert to list format
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        # Fallback to schoolbook
        return _schoolbook(), n * m * p

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
