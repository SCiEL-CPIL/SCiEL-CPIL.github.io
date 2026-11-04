"""Matrix multiplication 3x3 * 3x6: rank-18 block-wise schoolbook decomposition."""
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # rank = n * p = 18 (schoolbook for each of the two 3x3 blocks)
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18

    # For each output row i (0..n-1) and each output col l (0..p-1),
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # We group by l-block: for l in 0..2 and l in 3..5, use schoolbook 3x3.
    # Term index: (l_block, i, j) -> col = l_block * n + j  (n=3)
    # Wait, let me think more carefully.
    #
    # C[i, l] = sum_{j=0}^{m-1} A[i,j] * B[j,l]
    # The multiplication tensor has entries T[i,j,l] = delta_{i,i'} delta_{j,j'} delta_{l,l'}
    # for (i', j', l').
    #
    # Rank-r decomposition: T[i,j,l] = sum_{r} U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r]
    #
    # Schoolbook: one term for each (i', j', l'):
    #   U[i'*m+j', r] = 1, V[j'*p+l', r] = 1, W[l'*n+i', r] = 1
    # That gives rank = n*m*p = 54.
    #
    # Block-wise: fix l' in a block of 3. For l' in {0,1,2}, do 3x3 schoolbook on (i,j).
    # That's 9 terms per block, 2 blocks = 18 terms.
    # For term (block b, i', j'):
    #   U[i'*m+j', r] = 1
    #   V[j'*p + l', r] = 1 for l' in block b (3 values of l')
    #   W[l'*n+i', r] = 1 for l' in block b (3 values of l')
    #
    # But wait - in a rank-1 term, V is a single row-indexed vector and W is a single row-indexed vector.
    # V[j'*p + l', r] for multiple l' means V has 1s at multiple positions in the same column r.
    # That's fine - V is an (m*p) x rank matrix, so V[j, r] can be 1 for multiple j in the same column r.
    # Similarly W[l'*n+i', r] = 1 for multiple l' in the same column r.
    #
    # Let's verify: sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l'*n+i', r]
    # For term (b, i'', j''): U[i*m+j, r] = delta_{i,i''} delta_{j,j''}
    #                         V[j'*p+l, r] = 1 if j'=j'' and l in block b else 0
    #                         W[l'*n+i', r] = 1 if l'=l and i'=i'' else 0
    # Product = delta_{i,i''} delta_{j,j''} * [j'=j'' and l in block b] * [l'=l and i'=i'']
    #         = 1 if i=i'', j=j'', j'=j'', l=l', i'=i'', l in block b, l'=l
    #         = 1 if i=i', j=j', l=l' and l in block b (which is always true for some b)
    # Sum over b: for each l, exactly one block contains l, so sum = 1. Correct!

    col = 0
    for b in range(2):  # two blocks of 3 columns each
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                for l in range(b * n, (b + 1) * n):
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
