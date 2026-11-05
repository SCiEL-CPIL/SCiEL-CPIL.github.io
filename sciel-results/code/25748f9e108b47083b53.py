"""Matrix multiplication 3x3 * 3x7 tensor decomposition."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook baseline
    rank_sb = n * m * p  # 63
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1

    best = (rank_sb, U_sb, V_sb, W_sb)

    # Try to find a lower-rank decomposition via structured search.
    # For matmul (n x m) * (m x p), the tensor lives in R^{n*m} x R^{m*p} x R^{p*n}.
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21.
    # Upper bound from schoolbook: 63.
    # Known results: for 3x3 * 3x7, the best known rank is around 40-45 range.
    # Let's try a structured approach.

    # Strategy: Use the fact that matmul C = A*B where A is 3x3, B is 3x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k,l] = delta_{i,l} * delta_{j,k} (in appropriate indexing)
    # Actually T[A_idx, B_idx, C_idx] = 1 if C_idx = (i,k) and A_idx=(i,j), B_idx=(j,k)

    # Let's try rank values from a reasonable lower bound up.
    # Lower bound: at least max(n*p, m*max(n,p)) ... actually just n*p = 21 is a lower bound.
    # But that's very optimistic. Let's try from ~30 upward.

    # A practical approach: use bilinear algorithm search.
    # For small dimensions, we can enumerate or use known constructions.

    # Known: Strassen-like for rectangular matrices.
    # For 3x3 * 3x3, Strassen gives rank 7.
    # For 3x3 * 3x7, we can split the 3x7 into a 3x3 and 3x4 (or 3x3 + 3x4).
    # 3x3*3x3: rank 7 (Strassen)
    # 3x3*3x4: ?
    # 3x3*3x3: rank 7
    # 3x3*3x1: rank 3 (just 3 multiplications per column)
    # 3x3*3x4: we can do 3x3*3x3 (rank 7) + 3x3*3x1 (rank 3) = rank 10?
    # But that's not quite right because the B matrix columns are shared.

    # Actually, for A (3x3) * B (3x7):
    # Split B into B1 (3x3) and B2 (3x4).
    # C1 = A * B1 (3x3), C2 = A * B2 (3x4).
    # C = [C1 | C2].
    # Rank for A*B1: at most 7 (Strassen for 3x3*3x3)
    # Rank for A*B2: at most 3*4 = 12 schoolbook, or can we do better?
    # For 3x3 * 3x4: split B2 into two 3x2 blocks.
    # 3x3 * 3x2: rank at most 6 schoolbook. Can we do better?
    # 3x3 * 3x1: rank 3.
    # 3x3 * 3x2: could be rank 5? (not sure of exact value)

    # Let's just try: total rank = rank(3x3*3x3) + rank(3x3*3x4)
    # If we use 7 + 12 = 19, that's already below 63!
    # But wait, the A matrix is the SAME in both products. So we need to share the A factors.
    # The tensor decomposition must represent the full product as a sum of rank-1 tensors.
    # C[i,k] = sum_c U[i,j,c] * V[j,k',c] * W[k,i,c] ... no, the factorization is:
    # C = sum_c (U[:,c] outer V[:,c]) applied to A and B.
    # More precisely: C[i,k] = sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the decomposition is: T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # where a indexes A entries (n*m), b indexes B entries (m*p), c indexes C entries (p*n).
    # T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # = 1 if j=j' and i=i' and k=k', else 0.

    # So we need: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 for all valid (i,j,k)
    # and 0 otherwise. But since (i,j,k) determines the product entry, and the tensor
    # is 1 exactly when the indices are consistent (they always are in the product),
    # actually T_{(i,j),(j,k),(k,i)} = 1 for all i,j,k (the "consistent" ones),
    # and T = 0 for inconsistent index combos. But in the tensor space, we only have
    # the n*m * m*p * p*n entries, and the consistent ones are exactly the n*m*p = 63
    # that appear in the product. The inconsistent ones are 0.

    # So the tensor is: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # and we want to write this as sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r with small R.

    # Key insight: we can group by i (rows of A / rows of C).
    # For fixed i: T_i = sum_{j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # = (sum_j e_{(i,j)}) ⊗ (identity in j,k) ⊗ e_{(.,i)}
    # Hmm, this doesn't factor nicely.

    # Alternative: group by j (the shared dimension).
    # For fixed j: T_j = sum_{i,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # This is a matrix product tensor for the j-th "slice".

    # Let me just try to construct a rank-40-ish solution using block decomposition.
    # Split into: A*B where B is 3x7 = B_left(3x3) + B_right(3x4)
    # For 3x3 * 3x3: use Strassen rank 7
    # For 3x3 * 3x4: use schoolbook rank 12
    # But they share the A factor, so total rank = 7 + 12 = 19? No...
    # The decomposition is additive: sum over r of U_r V_r W_r.
    # If the 3x3*3x3 part uses ranks 1..7 and 3x3*3x4 uses ranks 8..19,
    # then for rank r in 1..7, W_r only has support on C columns 0..2 (k=0,1,2),
    # and for rank r in 8..19, W_r only has support on C columns 3..6 (k=3,4,5,6).
    # And U, V factors are consistent. This should work!

    # Let's build it:
    # Part 1: 3x3 * 3x3 (columns 0,1,2 of B and C)
    # Strassen for 3x3:
    # M1 = (A00+A11+A22)*(B00+B11+B22)
    # M2 = (A00-A11)*(B11-B22)
    # M3 = (A00-A22)*(B22-B00)
    # M4 = (A11+A22-A00)*(B00+B11-B22)
    # M5 = (A22+A00-A11)*(B11+B22-B00)
    # M6 = (A00+A11-A22)*(B22+B00-B11)
    # M7 = (A11+A22-A00-A11+... ) hmm let me look this up.

    # Actually, let me just use a known 7-term Strassen for 3x3:
    # M1 = (a+d+g)(p+s+w)
    # M2 = (a-d)(s-w)
    # M3 = (a-g)(w-p)
    # M4 = (d+g-a)(p+s-w)  ... I'm not confident in the exact formulas.

    # Given the complexity and risk of errors, let me just try a few structured
    # decompositions and verify them.

    # Simplest improvement: split B into 3x3 + 3x4.
    # 3x3*3x3 schoolbook: rank 9
    # 3x3*3x4 schoolbook: rank 12
    # Total: rank 21. That's the lower bound!
    # But can we actually achieve rank 21? Only if the two parts don't interact.
    # Since they use disjoint C columns, yes!

    # Let's build rank-21:
    rank21 = 21
    U21 = [[0]*rank21 for _ in range(n*m)]
    V21 = [[0]*rank21 for _ in range(m*p)]
    W21 = [[0]*rank21 for _ in range(p*n)]

    col = 0
    # Part 1: columns k=0,1,2 (3x3 * 3x3)
    for i in range(n):
        for j in range(m):
            for k in range(3):  # k=0,1,2
                U21[i*m+j][col] = 1
                V21[j*p+k][col] = 1
                W21[k*n+i][col] = 1
                col += 1
    # Part 2: columns k=3,4,5,6 (3x3 * 3x4)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):  # k=3,4,5,6
                U21[i*m+j][col] = 1
                V21[j*p+k][col] = 1
                W21[k*n+i][col] = 1
                col += 1

    # Verify rank-21
    def verify(U, V, W, rank):
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Reconstruct tensor
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check: for consistent (i,j,k), T[(i,j),(j,k),(k,i)] should be 1
        # For inconsistent, should be 0
        n_, m_, p_ = n, m, p
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    a = i * m_ + j
                    b = j * p_ + k
                    c = k * n_ + i
                    val = 0
                    for r in range(rank):
                        val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some inconsistent entries
        for i in range(n_):
            for j in range(m_):
                for j2 in range(m_):
                    if j2 == j:
                        continue
                    for k in range(p_):
                        a = i * m_ + j
                        b = j2 * p_ + k
                        c = k * n_ + i
                        val = 0
                        for r in range(rank):
                            val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                        if abs(val) > 1e-9:
                            return False
        return True

    if verify(U21, V21, W21, rank21):
        best = (rank21, U21, V21, W21)
    else:
        print("Rank-21 verification failed, falling back to schoolbook")

    # Now try to improve below 21 using Strassen on the 3x3 part.
    # Replace the 9 schoolbook terms for 3x3*3x3 with 7 Strassen terms.
    # Total rank: 7 + 12 = 19.

    # Strassen 3x3 (I'll use a known 7-term formula):
    # For matrices A and B (both 3x3), C = A*B:
    # M1 = (a + d + g) * (p + s + w)
    # M2 = (a - d) * (s - w)
    # M3 = (a - g) * (w - p)
    # M4 = (d + g - a) * (p + s - w)
    # M5 = (g + a - d) * (s + w - p)
    # M6 = (a + d - g) * (w + p - s)
    # M7 = (d + g - a - d + g - a + a - d) * ... this is getting messy.

    # Let me use a different approach: just try rank values and verify.
    # For now, return the best we have.

    _, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
