import time
import itertools

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 36

# Known rank-14 decomposition for 2x3x3 matrix multiplication.
# The decomposition is:
#   C_{ik} = sum_{r=1}^{14} U_{i,k? , r} * V_{?, r} * W_{?, r}
# But we need to be careful about the indices.
# The standard rank-14 decomposition for 2x3x3 is not trivially available in simple form.
# Let's use a simpler known decomposition for 2x2x2 (rank 7) and build up.
# 2x3x6 can be viewed as 2x3x3 + 2x3x3 if we split the columns of the second matrix.
# But we need a decomposition for 2x3x3.

# Let's try to find a rank-14 decomposition for 2x3x3.
# Actually, the rank of 2x3x3 is known to be 14 (from literature, e.g., Landsberg).
# We can try to construct it by taking the rank-7 decomposition of 2x2x2 and adding terms for the remaining parts.

# Let's use a known rank-14 decomposition for 2x3x3.
# The decomposition can be found in literature. Let's try to hardcode it.

# U: shape (2*3, 14) = (6, 14)
# V: shape (3*3, 14) = (9, 14)
# W: shape (3*2, 14) = (6, 14)

# Let's try to find a simple decomposition.
# We can use the fact that 2x3x3 can be decomposed into:
#   2x2x2 (rank 7) + 2x1x1 (rank 2) + 1x2x1 (rank 2) + 1x1x2 (rank 2) + ... 
# But this is getting complicated.

# Let's use a different approach: use the known rank-14 decomposition from the literature.
# The decomposition is given by:
#   U = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
#        [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
#        [0,0,0,1,0,0,0,0,0,0,0,0,0,0],
#        [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
#        [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
#        ...]
# This is not the right form.

# Let's try a different approach: use the rank-7 decomposition for 2x2x2 and extend it.
# 2x3x3 = 2x2x2 + 2x1x1 + 1x2x1 + 1x1x2
# The 2x2x2 part has rank 7.
# The 2x1x1 part has rank 2 (since it's just 2 scalar multiplications).
# The 1x2x1 part has rank 2.
# The 1x1x2 part has rank 2.
# Total rank = 7 + 2 + 2 + 2 = 13. But we need 14.

# Actually, the decomposition is not quite that simple.

# Let's use a known rank-14 decomposition for 2x3x3.
# We can find it in the literature or derive it.

# For now, let's use the schoolbook decomposition for 2x3x6, which has rank 36.
# This is safe and correct.

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
