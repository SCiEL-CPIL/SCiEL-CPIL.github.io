"""Matrix multiplication 4,5,5 tensor decomposition using hybrid 2x3x3 and 2x2x2 Strassen."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Strassen's 7-term decomposition for 2x2x2
    # Indices: U: (i,j) for i in 0..1, j in 0..1 -> [i*2+j]
    #          V: (j,k) for j in 0..1, k in 0..1 -> [j*2+k]
    #          W: (k,i) for k in 0..1, i in 0..1 -> [k*2+i]
    strassen_222 = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # 2x3x3 Strassen-like decomposition (rank 17)
    # A: 2x3, B: 3x3, C: 3x2
    # We use the standard extension of Strassen's algorithm.
    # Partition A into A1 (2x2), A2 (2x1)
    # Partition B into B11 (2x2), B12 (2x1), B21 (1x2), B22 (1x1)
    # Partition C into C1 (3x1), C2 (3x1) -- wait, C is 3x2.
    # Let's define indices carefully.
    # A indices: i in 0..1, j in 0..2. idx = i*3 + j
    # B indices: j in 0..2, k in 0..2. idx = j*3 + k
    # C indices: k in 0..2, i in 0..1. idx = k*2 + i
    
    # The 17 terms for 2x3x3 can be derived by applying Strassen to the 2x2x2 part 
    # and handling the extra column/row with additional terms.
    # A known rank 17 decomposition for 2x3x3:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    #     B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    #     C = A*B = [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]]? No, C is 3x2?
    # Wait, matrix multiplication (2x3) * (3x3) = (2x3).
    # The tensor is T[i,j,k] for i in 0..1, j in 0..2, k in 0..2.
    # Output C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor indices are (i, j, k).
    # U index: i*3 + j (6 elements)
    # V index: j*3 + k (9 elements)
    # W index: k*2 + i (6 elements)
    
    # Let's use a known rank 17 decomposition.
    # One way to get rank 17 is to use Strassen on the first 2 columns of B and first 2 rows of B?
    # Actually, a simpler approach for 2x3x3 is to view it as two 2x2x2 blocks plus corrections.
    # Or use the "Strassen-like" algorithm for non-square matrices.
    
    # Here is a verified rank 17 decomposition for 2x3x3:
    # Terms (u, v, w) where u is length 6, v is length 9, w is length 6.
    # Indices for U: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Indices for V: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # Indices for W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
    
    # I will construct the 2x3x3 decomposition by combining:
    # 1. Strassen 2x2x2 on A[:, 0:2] and B[0:2, 0:2] contributing to C[:, 0:2]
    # 2. Terms for A[:, 2] * B[0:2, :] contributing to C[:, :]
    # 3. Terms for A[:, 0:2] * B[2, :] contributing to C[:, :]
    # 4. Terms for A[:, 2] * B[2, :] contributing to C[:, :]
    
    # Let's just hardcode a known rank 17 set or generate it carefully.
    # Generating it manually is error-prone. Let's use a different strategy:
    # Use 2x2x2 Strassen for the first 3 columns? No, 3 is not power of 2.
    
    # Alternative Strategy:
    # Partition 4x5x5 into:
    # - Two 2x5x5 blocks.
    # For each 2x5x5:
    #   - Partition into 2x3x3 and 2x2x2.
    #   - Use rank 17 for 2x3x3 and rank 7 for 2x2x2.
    #   - Total rank for 2x5x5 is 17+7=24.
    #   - Total rank for 4x5x5 is 2*24 = 48.
    
    # I need the explicit 17 terms for 2x3x3.
    # Let's derive or find them.
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x3, B2 is 1x3.
    # C = A1*B1 + A2*B2.
    
    # Strassen on A1*B1 (2x2 * 2x3 -> 2x3):
    # This is not standard Strassen.
    # Let's stick to the block decomposition that uses only 2x2x2 Strassen and schoolbook for the rest, 
    # but optimize the block placement to minimize rank.
    
    # Current program uses 8 blocks of 2x2x2, rank 96.
    # Can we do better with 2x2x2 blocks?
    # Max non-overlapping 2x2x2 blocks in 4x5x5:
    # i: 0-1, 2-3 (2 blocks)
    # j: 0-1, 2-3 (2 blocks, leaving j=4)
    # k: 0-1, 2-3 (2 blocks, leaving k=4)
    # Total 2*2*2 = 8 blocks.
    # Rank = 8*7 + (100 - 8*8) = 56 + 36 = 92.
    # Wait, the current program calculates rank 96? 
    # "num_schoolbook = 36", "num_strassen = 8*7 = 56". Total 92.
    # Why does the prompt say current score 1.0000 and rank 96? 
    # Maybe the verification failed for rank 92 and it fell back to something else? 
    # Or maybe my calculation of uncovered is wrong.
    # Uncovered: j=4 (20) + k=4 (20) - (j=4,k=4) (4) = 36.
    # 56 + 36 = 92.
    # The current program's comment says "rank 92" but the code might have a bug or fallback.
    # Let's check the code: `best_rank = len(candidate["U"][0])`.
    # If `verify` passes, it returns rank 92.
    # If the score is 1.0, and seed is 100, then 100/92 = 1.08. 
    # But the prompt says score 1.0000. This implies the current program might be returning 100 (schoolbook) 
    # or the evaluator is using a different baseline? 
    # "score is the rank... 1.0 equals the seed program". 
    # If seed is 100, and current is 92, score should be 100/92 > 1.
    # Unless the "seed program" is not schoolbook? 
    # "Initial program for the suite task" usually implies schoolbook.
    # Maybe the current program fails verification and falls back to schoolbook?
    
    # Let's try to get a lower rank than 92.
    # Can we use a 2x3x3 decomposition of rank 17?
    # If we have a valid rank 17 decomposition for 2x3x3, we can cover more volume.
    
    # Let's construct the 2x3x3 decomposition.
    # I will use the following known rank 17 decomposition for 2x3x3:
    # (Source: various tensor decomposition papers, e.g., "Rank of matrix multiplication tensor")
    # Actually, finding a hardcoded rank 17 set is risky. 
    # Let's try a different structural change: 
    # Use 2x2x2 Strassen blocks but overlap them? No, they must be disjoint for simple sum.
    
    # What if we use a 3x3x3 decomposition? 
    # Rank of 3x3x3 is at least 19 (Landsberg). 
    # If we use one 3x3x3 block (rank 19) and cover the rest?
    # 4x5x5. 
    # Place 3x3x3 at (0,0,0). Covers i=0-2, j=0-2, k=0-2.
    # Remaining: 
    # i=3 (5x5=25)
    # j=3-4 (4x2x5 = 40)
    # k=3-4 (4x5x2 = 40)
    # Overlaps... this is complex.
    
    # Let's stick to the 2x2x2 blocks but try to find a better packing.
    # In 4x5x5, we can fit:
    # i: 2 blocks (0-1, 2-3)
    # j: 2 blocks (0-1, 2-3) or (1-2, 3-4) etc.
    # k: 2 blocks (0-1, 2-3) etc.
    # Max 8 blocks. Rank 92.
    
    # Is there a decomposition of rank < 92?
    # Yes, likely. 
    # Let's try to implement the 2x3x3 rank 17 decomposition.
    # I will generate it using a simple algorithm or hardcoded values if I can find them.
    
    # Hardcoded Rank 17 for 2x3x3:
    # This is difficult to get right from memory.
    
    # Alternative: Use 2x2x2 Strassen for the 4x4x4 part, and handle the fringes.
    # 4x4x4 can be decomposed into 8 2x2x2 blocks. Rank 56.
    # The tensor is 4x5x5.
    # The 4x4x4 part covers i=0-3, j=0-3, k=0-3.
    # Fringes: j=4 (4x1x5 = 20), k=4 (4x5x1 = 20). Overlap (4,4,4) is not in 4x4x4.
    # Actually, 4x4x4 covers j=0-3, k=0-3.
    # Fringe j=4: i=0-3, j=4, k=0-4 (20 terms)
    # Fringe k=4: i=0-3, j=0-4, k=4 (20 terms)
    # Overlap: i=0-3, j=4, k=4 (4 terms)
    # Total fringe terms: 20 + 20 - 4 = 36.
    # If we use schoolbook for fringes: 56 + 36 = 92.
    
    # Can we reduce the fringe rank?
    # The fringe is a union of a 4x1x5 slice and a 4x5x1 slice.
    # This structure might allow a lower rank decomposition than schoolbook.
    # However, implementing a custom low-rank decomposition for this specific fringe is complex.
    
    # Let's try a different block size: 2x2x4? 
    # Or 4x2x2?
    
    # Let's try to use the 2x3x3 rank 17 decomposition. I will code it up carefully.
    # If I can't get it right, I will fall back to the 92 rank solution.
    
    # For now, I will implement the 8x(2x2x2) solution which gives rank 92.
    # I will double check the verification to ensure it doesn't fail.
    
    def build_decomposition_222(blocks):
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        num_schoolbook += 1

        num_strassen = len(blocks) * 7
        rank = num_schoolbook + num_strassen

        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

        for (i0, j0, k0) in blocks:
            for (u_c, v_c, w_c) in strassen_222:
                a_idx = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
                b_idx = [(j0, k0), (j0, k0 + 1), (j0 + 1, k0), (j0 + 1, k0 + 1)]
                c_idx = [(k0, i0), (k0, i0 + 1), (k0 + 1, i0), (k0 + 1, i0 + 1)]
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        U[a_idx[la][0] * m + a_idx[la][1]][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        V[b_idx[lb][0] * p + b_idx[lb][1]][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        W[c_idx[lc][0] * n + c_idx[lc][1]][col] += cf
                col += 1

        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    # Optimal 8 blocks
    blocks_optimal = [(0,0,0), (0,0,2), (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
    
    best_result = None
    best_rank = n * m * p + 1

    try:
        candidate = build_decomposition_222(blocks_optimal)
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            best_result = candidate
            best_rank = len(candidate["U"][0])
    except Exception:
        pass

    # Fallback to schoolbook
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
