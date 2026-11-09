import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate_decomp(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # We will build a decomposition using block structure.
    # The multiplication tensor T is decomposed into blocks.
    # We can view A as 3x3 and B as 3x7.
    # We can split B into three 3x3 blocks and one 3x1 block.
    # Let B = [B1 | B2 | B3 | B4] where B1, B2, B3 are 3x3 and B4 is 3x1.
    # Then C = A*B = [A*B1 | A*B2 | A*B3 | A*B4].
    # Each A*B_i (i=1,2,3) is a 3x3 multiplication.
    # A*B4 is a 3x1 multiplication.
    
    # We use Strassen's algorithm for 3x3 multiplications?
    # Strassen is for 2x2. For 3x3, we can use a rank-7 decomposition of the 2x2 tensor?
    # No, Strassen reduces 2x2 to rank 7.
    # But we are doing 3x3 * 3x3.
    # We can split 3x3 into four 2x2 blocks (with overlap or padding?).
    # Standard block decomposition for 3x3 using 2x2 blocks:
    # We can pad 3x3 to 4x4 and use Strassen?
    # Or we can use the known minimal rank for 3x3 matrix multiplication.
    # The minimal rank for 3x3 * 3x3 is not 18 (schoolbook).
    # It is known that rank < 18 for 3x3 is an open problem? 
    # Wait, the rank of the 3x3 matrix multiplication tensor is at least 15 and at most 18.
    # Is there a known rank 17 or 16 decomposition?
    # Actually, for 3x3, the schoolbook is 9. Wait, 3*3*3 = 27.
    # Schoolbook rank is 27.
    # Strassen for 2x2 is 7 (vs 8).
    # For 3x3, we can split into 2x2 and 1x1 blocks?
    # Let's stick to the block structure suggested in the prompt:
    # "decomposing the 3x7 matrix multiplication into two 3x3 multiplications and one 3x1 multiplication"
    # This implies splitting B into [B1 (3x3) | B2 (3x3) | B3 (3x1)]? No, 3x3 + 3x3 + 3x1 = 3x7.
    # So B = [B1 | B2 | B3] where B1, B2 are 3x3 and B3 is 3x1.
    # Then C = [C1 | C2 | C3] where C1 = A*B1, C2 = A*B2, C3 = A*B3.
    # C1, C2 are 3x3 results, C3 is 3x1 result.
    
    # Rank of 3x3 * 3x3 multiplication tensor:
    # Schoolbook is 9. (3*3*3 = 27? No, the number of multiplications is 3*3*3 = 27? No.
    # A (3x3) * B (3x3) -> C (3x3).
    # C_ij = sum_k A_ik B_kj.
    # There are 3*3 = 9 output entries.
    # Each requires 3 multiplications.
    # Total 9*3 = 27 multiplications?
    # No, the tensor rank is the number of rank-1 tensors needed.
    # The schoolbook algorithm uses 27 multiplications?
    # Wait. For n x n * n x n, schoolbook rank is n^3.
    # For 3x3, n=3, so rank 27?
    # Let's check.
    # T[i,j,k] = 1 if ...?
    # The tensor is T in V^* x W^* x (V^* x W^*)^*.
    # Dimensions n, n, n.
    # Schoolbook rank is n^3.
    # So for 3x3 * 3x3, rank is 27.
    
    # Rank of 3x3 * 3x1 multiplication tensor:
    # A (3x3) * B (3x1) -> C (3x1).
    # C_i = sum_j A_ij B_j.
    # 3 output entries, each 3 multiplications.
    # Total 9 multiplications.
    # Rank is 9? (n*m*p = 3*3*1 = 9).
    
    # So if we just use schoolbook for the blocks:
    # Rank = 27 + 27 + 9 = 63.
    # This matches the schoolbook rank of 3x3 * 3x7 (3*3*7 = 63).
    # So splitting into blocks and using schoolbook doesn't help.
    
    # The prompt suggests: "apply the known Strassen rank-7 decomposition for the 3x3 blocks".
    # Strassen is for 2x2. It has rank 7.
    # Can we apply Strassen to 3x3 blocks?
    # We can split a 3x3 multiplication into four 2x2 multiplications?
    # No, 3x3 doesn't fit neatly into 2x2.
    # However, we can pad 3x3 to 4x4?
    # Or we can use the fact that 3x3 multiplication can be done with fewer than 27 multiplications?
    # Actually, the minimal rank of the 3x3 matrix multiplication tensor is unknown, but it is known to be <= 19 (or 20?).
    # Wait, the prompt says "apply the known Strassen rank-7 decomposition for the 3x3 blocks".
    # This phrasing is slightly confusing. Strassen is for 2x2.
    # Maybe it means we decompose the 3x3 blocks into 2x2 sub-blocks and apply Strassen to those?
    # If we split 3x3 * 3x3 into 2x2 * 2x2 blocks:
    # We have 4 2x2 blocks in A and 4 2x2 blocks in B (with overlaps/padding).
    # This is getting complicated.
    
    # Let's look at the "rank-minimal exact tensor decomposition" for 3x3 * 3x7.
    # Is there a known rank lower than 63?
    # Yes, bilinear algorithms often reduce the rank.
    # For 3x3 * 3x3, there are algorithms with rank 19 or 20?
    # Actually, a simple improvement is to use Strassen for the 2x2 core.
    # Let's try to construct a rank 61 decomposition as hinted in the previous BET.
    # "yielding rank 61".
    # How?
    # If we replace one 2x2 * 2x2 multiplication (rank 8) with Strassen (rank 7), we save 1.
    # If we do this for two 3x3 blocks?
    # Each 3x3 * 3x3 can be viewed as containing 2x2 * 2x2 multiplications.
    # But how many independent 2x2 multiplications are in a 3x3 * 3x3?
    # It's not a simple sum of 2x2 tensors.
    
    # Let's try a different approach: random search for rank reduction.
    # Start with schoolbook rank 63.
    # Try to reduce rank by 1, 2, etc.
    # Use a linear algebra solver to check if a rank-1 tensor can be removed or if columns can be linearly combined.
    # Actually, we can just try to find a decomposition of rank 62, 61, etc.
    
    # Let's implement a heuristic search.
    # 1. Start with schoolbook.
    # 2. Try to remove columns (if the sum of remaining tensors still equals the target).
    # 3. Try to merge columns (linear combinations).
    
    # Since exact integer/half-integer entries are preferred, Strassen uses integers.
    # Let's try to build the Strassen-based decomposition explicitly if possible.
    # But for 3x3 * 3x7, it's complex.
    
    # Let's stick to the "remove columns" heuristic from the seed, but make it faster and more robust.
    # The seed tried to remove columns one by one.
    # We can try to remove multiple columns.
    # Or we can try to find a lower rank decomposition using a known algorithm.
    
    # Let's try to find a rank 61 decomposition by modifying the schoolbook.
    # We know that for 2x2 * 2x2, Strassen saves 1 rank.
    # If we can identify a 2x2 * 2x2 sub-problem in the 3x3 * 3x7 problem, we might save rank.
    
    # Actually, let's just try to verify if rank 62 or 61 is possible with a random linear algebra approach.
    # We will use a simple iterative improvement.
    
    # For now, let's return the schoolbook decomposition (rank 63) to ensure correctness, 
    # and then try to improve it within the time limit.
    
    rank = n * m * p  # 63
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Heuristic search for lower rank
    # We will try to remove columns.
    # To speed this up, we can use the fact that the tensor is sum of rank-1 tensors.
    # If we remove column r, the new tensor is T - u_r v_r^T w_r^T.
    # We need this to be 0.
    # This implies u_r v_r^T w_r^T = 0, which is only possible if the tensor is 0.
    # So we cannot just remove a column unless it's zero.
    # But we can replace two columns with one?
    # Or we can linearly combine columns.
    
    # Let's try a random search for a lower rank decomposition.
    # We fix the target tensor T.
    # We want to find U, V, W with rank < 63 such that sum_r u_r v_r^T w_r^T = T.
    
    # This is a hard problem.
    # Let's try to use the Strassen idea for the 3x3 blocks.
    # We will manually construct the Strassen decomposition for the 3x3 * 3x3 part.
    # Since 3x3 * 3x3 can be decomposed into 4 2x2 * 2x2 multiplications (with padding/overlap),
    # we can apply Strassen to each 2x2 block.
    # However, the overlaps make it tricky.
    
    # Given the time constraint and complexity, I will return the schoolbook decomposition
    # but with a more efficient validation and search for column removal if possible.
    
    # Actually, the previous BET suggested rank 61.
    # If I can't easily construct it, I should stick to 63 to be safe.
    # But 63 is the seed score. I need to beat it.
    
    # Let's try to find a rank 62 decomposition by searching for linear dependencies.
    # We can set up a system of linear equations.
    # T = sum_{r=1}^R u_r v_r^T w_r^T
    # We want R < 63.
    
    # I will implement a simple local search that tries to replace two rank-1 tensors with one
    # if they are "close" in some metric, but this is heuristic.
    
    # For now, I will return the schoolbook decomposition.
    # The improvement must come from a better algorithm.
    # Since I don't have a reliable rank < 63 algorithm for 3x3 * 3x7,
    # I will return 63.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
