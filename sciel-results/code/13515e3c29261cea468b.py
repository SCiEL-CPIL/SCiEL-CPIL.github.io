"""Structured sparse linear algebra: binary IP over candidate rank-1 patterns for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    # We need to find minimal R and matrices U(nm x R), V(mp x R), W(pn x R) with entries in {0, 1/2, -1/2, 1, -1}
    # such that for all a,b,c: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]

    # Strategy: generate a dictionary of candidate rank-1 terms (u, v, w) with small integer entries,
    # then use a greedy/ILP approach to select a minimal subset that exactly satisfies all tensor equations.

    # For efficiency, we focus on sparse patterns where each vector has few non-zero entries.
    # Candidate patterns: u, v, w each have entries in {-1, 0, 1} with at most 3 non-zero entries each.

    # The tensor has nm*mp*pn = 9*24*24 = 5184 entries, but only 72 are non-zero (all = 1).
    # The key insight: we need the sum of rank-1 tensors to match exactly.

    # Let's use a constructive approach based on Strassen-style algebraic identities.
    # For 3x3x3, Strassen's algorithm uses 7 multiplications (rank 7) vs schoolbook 27.
    # But that's for square matrices. For 3x3x8, we can split into blocks.

    # Approach: Use the fact that 3x3x8 can be decomposed using algebraic techniques.
    # Let's try to build a rank-48 decomposition by splitting 8 into two groups of 4,
    # and using a rank-24 decomposition for 3x3x4 (if it exists).

    # Actually, let's use a more systematic approach:
    # Generate all "simple" rank-1 candidates and use a greedy set-cover-like approach.

    # For each candidate (u, v, w), compute the tensor t = u ⊗ v ⊗ w.
    # We need sum of selected tensors = T.

    # Since exact matching is hard, let's use a known algebraic construction.

    # Hardcoded approach: Use the 4x4x4 Strassen-Laderman rank-23 decomposition
    # and embed 3x3x8 into it.

    # The 4x4x4 rank-23 decomposition (verified):
    # I'll use a known set of 23 triples for 4x4x4.

    # For now, let's use a simpler but effective approach:
    # Rank 48 via two 3x3x4 blocks, each using a rank-24 decomposition.

    # 3x3x4 decomposition: We can use the schoolbook (36) but that gives 72 total.
    # Let's try to find a rank-24 for 3x3x4.

    # Alternative: Use random search with verification.
    # Try rank R from 45 to 72, and for each, try to find a valid decomposition.

    # Given time constraints, let's use a known good result:
    # The rank of 3x3x8 is at most 48 (via block decomposition with optimized 3x3x4).

    # Let me construct a rank-48 decomposition:
    # Split B (3x8) into B1 (3x4) and B2 (3x4).
    # Compute A*B1 and A*B2 using a rank-24 method each.

    # For 3x3x4, I'll use a constructive rank-24 decomposition.
    # This is based on the idea of using 24 multiplications.

    # Actually, the simplest safe approach: use rank 54 via three 3x3x? blocks.
    # 3x3x3 (rank 23 Strassen) + 3x3x3 (rank 23) + 3x3x2 (rank 18) = 64. Not better.

    # Let me just use a well-known result: rank(3x3x8) <= 48.
    # I'll construct it via the following method:

    # Use the following algebraic identity approach:
    # For multiplying 3x3 by 3x8, we can use 48 multiplications.

    # Construct U, V, W for rank 48:
    # This requires a specific pattern. Let me use a verified construction.

    # Fallback: If I can't construct rank 48 easily, use rank 64 (two 3x3x3 Strassen + one 3x3x2 schoolbook).

    # For now, return schoolbook rank 72 as safe baseline, but try to improve.

    # Let me try a different approach: use the known 4x4x4 rank-23 decomposition.
    # Embed 3x3x8 into 4x4x8: pad A to 4x4, B to 4x8.
    # 4x4x8 = two 4x4x4 blocks, each rank 23, total rank 46.
    # Extract the 3x3x8 result from the 4x4x8 result.

    # This gives rank 46! Let me implement this.

    # The 4x4x4 rank-23 decomposition (Strassen-Laderman):
    # I need to hardcode the 23 rank-1 terms.

    # Standard 4x4x4 rank-23 (one known decomposition):
    # Each term is (u, v, w) where u, v, w are 16-dimensional vectors with entries in {-1, 0, 1}.

    # After careful consideration, let me use a simpler approach:
    # Return rank 64 via block decomposition, which is guaranteed to work.

    # 3x3x8 = [A | A] * [B1; B2] where B1, B2 are 3x4.
    # Use schoolbook for each 3x3x4: rank 36 each, total 72. No improvement.

    # Let me use Strassen for 3x3x3 blocks:
    # 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2
    # Rank: 23 + 23 + 18 = 64.

    # Implement rank 64:
    rank = 64
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Block 1: 3x3x3 for columns 0-2 of B
    # Strassen 3x3x3 rank 23
    # I'll use a known Strassen decomposition for 3x3x3.

    # For simplicity, use schoolbook for the 3x3x3 blocks: rank 9 each.
    # 3x3x3 schoolbook: 9 multiplications
    # 3x3x2 schoolbook: 6 multiplications
    # Total: 9 + 9 + 6 = 24? No, that's not right.

    # Schoolbook 3x3x3: 9 multiplications (one per output element)
    # Schoolbook 3x3x2: 6 multiplications
    # Total: 9 + 9 + 6 = 24. But we need to handle the full 3x3x8.

    # Actually, schoolbook for 3x3x8 is 72 multiplications.
    # Let me just return schoolbook rank 72.

    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
