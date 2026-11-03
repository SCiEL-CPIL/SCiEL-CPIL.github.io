"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W):
    """Check if U, V, W form a valid exact decomposition for the 2x3 @ 3x8 tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check if entries are integers or half-integers
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
            return False
    # Compute the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare to 1 if i,j,k are in the tensor (which is always 1 for valid indices)
    # Actually, the tensor T[i,j,k] = 1 if the indices are valid, else 0.
    # But here all indices are valid, so T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    # Let's compute this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _make_valid(U, V, W):
    """Snap entries to nearest half-integer and return as lists."""
    U = np.round(np.array(U) * 2) / 2
    V = np.round(np.array(V) * 2) / 2
    W = np.round(np.array(W) * 2) / 2
    return U.tolist(), V.tolist(), W.tolist()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    best_rank = n * m * p
    best_result = None
    
    # First, return schoolbook if we run out of time
    def schoolbook():
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
    
    if best_result is None:
        best_result = schoolbook()
    
    # Try to find a better decomposition using a constructive approach.
    # The tensor is M_{2,3,8}. We can try to exploit structure.
    # One approach: use the fact that 8 = 2*4, so we can write it as 2 blocks of 2x3x4.
    # If we can find a rank-20 decomposition for 2x3x4, then total rank is 40.
    # But 2x3x4 has schoolbook rank 24. Can we do better?
    # The tensor M_{2,3,4} has 24 non-zero entries.
    # Let's try to find a decomposition of rank < 24 for 2x3x4 using a simple search.
    
    # Actually, let's try a different approach: use a randomized search with gradient-like updates.
    # But that's complex. Instead, let's try a small number of hand-crafted decompositions.
    
    # Try rank 40: two 2x3x4 blocks, each with rank 20.
    # We need to find U1, V1, W1 of shapes (6,20), (12,20), (8,20) such that
    # sum_r U1[i*3+j, r] * V1[j*4+k, r] * W1[k*2+i, r] = 1 for all i,j,k in the 2x3x4 tensor.
    # And similarly for the second block.
    
    # Let's try to use a known result: the rank of M_{2,3,4} is at most 22 (I think).
    # But we don't have a concrete decomposition.
    
    # Instead, let's try a simple heuristic: start with schoolbook and try to zero out entries
    # by finding linear dependencies.
    
    # Given the time constraints, let's just return the schoolbook decomposition.
    # The previous program had score 1.0, which is the schoolbook rank 48.
    # If we can find a rank < 48, we improve.
    
    # Let's try a quick randomized search for a rank 44 decomposition.
    # We'll use a simple iterative method: start with random matrices and do a few iterations.
    
    # Actually, let's try a constructive approach based on the following idea:
    # For M_{2,3,8}, we can group the 8 columns into 4 groups of 2.
    # For each group, we have a 2x3x2 sub-tensor.
    # The rank of M_{2,3,2} is at most 12 (schoolbook).
    # But maybe we can do better for 2x3x2.
    
    # Let's try to find a decomposition of rank 10 for 2x3x2.
    # If we can, then total rank is 4*10 = 40.
    
    # Let's do a small search for 2x3x2.
    # The tensor has 12 non-zero entries.
    # We need U (6xR), V (6xR), W (4xR) such that sum_r U[i*3+j,r] V[j*2+k,r] W[k*2+i,r] = 1.
    
    # Let's try R=10.
    if time.time() < deadline:
        # Search for a decomposition of M_{2,3,2} with rank 10.
        # We'll use a simple randomized approach with local search.
        for trial in range(100):
            if time.time() >= deadline:
                break
            R = 10
            random.seed(trial)
            U = np.random.randn(6, R) * 0.5
            V = np.random.randn(6, R) * 0.5
            W = np.random.randn(4, R) * 0.5
            # Do a few gradient steps
            for _ in range(100):
                if time.time() >= deadline:
                    break
                # Compute error
                err = 0.0
                gradU = np.zeros_like(U)
                gradV = np.zeros_like(V)
                gradW = np.zeros_like(W)
                for i in range(2):
                    for j in range(3):
                        for k in range(2):
                            s = np.sum(U[i*3+j] * V[j*2+k] * W[k*2+i])
                            e = s - 1.0
                            err += e * e
                            gradU[i*3+j] += 2 * e * V[j*2+k] * W[k*2+i]
                            gradV[j*2+k] += 2 * e * U[i*3+j] * W[k*2+i]
                            gradW[k*2+i] += 2 * e * U[i*3+j] * V[j*2+k]
                lr = 0.01
                U -= lr * gradU
                V -= lr * gradV
                W -= lr * gradW
            # Check if we can snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            # Verify
            valid = True
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        s = np.sum(U_snap[i*3+j] * V_snap[j*2+k] * W_snap[k*2+i])
                        if not np.isclose(s, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Great! We found a rank-10 decomposition for 2x3x2.
                # Now construct the full decomposition for 2x3x8.
                # Use 4 copies of this decomposition.
                total_rank = 4 * R
                U_full = np.zeros((6, total_rank))
                V_full = np.zeros((6, total_rank))  # Wait, V shape is (m*p, R) = (3*8, R) = (24, R)
                # Actually, V has shape (m*p, R) = (24, R). For each block of 2 columns, V has 6 entries.
                # So for 4 blocks, V has 4*6 = 24 entries.
                # Similarly, W has shape (p*n, R) = (16, R). For each block, W has 4 entries.
                # So for 4 blocks, W has 4*4 = 16 entries.
                # Let's reconstruct properly.
                U_full = np.zeros((n*m, total_rank))  # (6, 40)
                V_full = np.zeros((m*p, total_rank))  # (24, 40)
                W_full = np.zeros((p*n, total_rank))  # (16, 40)
                for block in range(4):
                    for r in range(R):
                        col = block * R + r
                        for i in range(2):
                            for j in range(3):
                                U_full[i*3+j, col] = U_snap[i*3+j, r]
                        for j in range(3):
                            for k in range(2):
                                # k in [0,2) corresponds to column block*2+k
                                V_full[j*8 + block*2 + k, col] = V_snap[j*2+k, r]
                        for k in range(2):
                            for i in range(2):
                                W_full[(block*2+k)*2 + i, col] = W_snap[k*2+i, r]
                # Verify the full decomposition
                if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                    best_rank = total_rank
                    best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                    break  # Found a good one, stop searching
    # Return the best result found
    return best_result
# EVOLVE-BLOCK-END
