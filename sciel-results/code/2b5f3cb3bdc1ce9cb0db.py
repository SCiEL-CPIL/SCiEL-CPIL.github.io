"""
Matrix multiplication 3x3x6: rank-minimal exact tensor decomposition.

The 3x3x6 product is two independent 3x3x3 products sharing the same A.
We use the Strassen-like rank-23 decomposition for 3x3x3 (Lader 1972)
and duplicate the U factor for the second block, giving total rank 46.

The key insight: the current seed program's Lader decomposition is NOT
actually verified to be correct (it likely fails the verification check
and falls back to schoolbook rank 54). We need a CORRECT rank-23
decomposition.

Actually, let me reconsider. The tensor rank of 3x3x3 matrix multiplication
is known to be 23 (Lader). But constructing an explicit valid decomposition
is non-trivial. Let me use a different approach.

For 3x3x6, we can think of it as:
- A is 3x3, B is 3x6, C is 3x6
- C[i,k] = sum_j A[i,j] * B[j,k]

The schoolbook rank is 3*3*6 = 54.

Can we do better? The two 3x3x3 blocks share the same A factor.
If we have a rank-r decomposition for 3x3x3:
  sum_{s=1}^{r} U_s[i,j] * V_s[j,k] * W_s[k,i] = 1

Then for 3x3x6:
  U_combined[i*3+j, s] = U_s[i,j] for s in [0,r) and [r,2r)
  V_combined[j*6+k, s] = V_s[j,k] for s in [0,r), k in [0,3)
  V_combined[j*6+k, s] = V_{s-r}[j,k-3] for s in [r,2r), k in [3,6)
  W_combined[k*3+i, s] = W_s[k,i] for s in [0,r), k in [0,3)
  W_combined[k*3+i, s] = W_{s-r}[k-3,i] for s in [r,2r), k in [3,6)

This gives rank 2r. For r=23, rank=46.

But we need a VALID rank-23 decomposition. The seed program's attempt
appears to be incorrect (it's a guess at Lader's algorithm that likely
doesn't satisfy the tensor identity).

Let me use a known valid construction. Actually, the simplest correct
approach is to use the schoolbook for each 3x3x3 block (rank 9 each),
giving total rank 18. Wait, that's not right either.

For 3x3x3 schoolbook: rank = 9 (one term per (i,j,k) triple).
For 3x3x6: two blocks, each rank 9, but they share A...

Actually no. Let me think again.

3x3x3 multiplication: C[i,k] = sum_j A[i,j] B[j,k]
Schoolbook: for each (i,j,k), one term. Total 9 terms.
U[i*3+j, (i,j,k)] = 1, V[j*3+k, (i,j,k)] = 1, W[k*3+i, (i,j,k)] = 1

For 3x3x6: C[i,k] = sum_j A[i,j] B[j,k], k in [0,5]
Schoolbook: 3*3*6 = 54 terms.

But we can split into two 3x3x3 blocks:
Block 1: k in [0,3), uses A, B1 (cols 0-2), C1 (cols 0-2)
Block 2: k in [3,6), uses A, B2 (cols 3-5), C2 (cols 3-5)

Each block is a 3x3x3 multiplication with the same A.
Schoolbook for each: 9 terms.
Total: 18 terms.

U[i*3+j, s] for s in [0,9): from block 1
U[i*3+j, s] for s in [9,18): from block 2 (same A values)

V[j*6+k, s] for k in [0,3), s in [0,9): V1[j*3+k, s]
V[j*6+k, s] for k in [3,6), s in [9,18): V2[j*3+(k-3), s-9]

W[k*3+i, s] for k in [0,3), s in [0,9): W1[k*3+i, s]
W[k*3+i, s] for k in [3,6), s in [9,18): W2[(k-3)*3+i, s-9]

This gives rank 18, which is better than schoolbook 54.

Can we do even better? The tensor rank of 3x3x3 is 23, so using
Strassen-like would give 23+23=46, which is worse than 18.

So the best simple approach is rank 18 using the block structure.

But wait - can we share more between the blocks? The A factor is the same,
but in the tensor decomposition, U encodes A's interaction. If we use
schoolbook for each block, U for block 1 has entries at positions (i,j)
for each (i,j,k) triple, and similarly for block 2. Since the (i,j) indices
are the same, we might be able to merge... but no, because the V and W
factors differ between blocks.

Actually, in the schoolbook decomposition, each term is indexed by (i,j,k).
For block 1: terms (i,j,k) with k in [0,3)
For block 2: terms (i,j,k) with k in [3,6)

The U factor for term (i,j,k) is 1 at position i*3+j.
So U[i*3+j, (i,j,k)] = 1.

For block 1, k in [0,3): U[i*3+j, s] = 1 where s corresponds to (i,j,k)
For block 2, k in [3,6): U[i*3+j, s] = 1 where s corresponds to (i,j,k)

These are different terms (different s), so they can't be merged.
Total rank = 9 + 9 = 18.

This is the best we can do with this simple block approach.

Let me verify: is there a way to get below 18?
The tensor rank of the 3x3x6 multiplication tensor... 
The 3x3x6 tensor has 3*3*6 = 54 entries (all 1s in the "diagonal" sense).
The rank is at least max(3*3, 3*6, 6*3) = max(9, 18, 18) = 18 by the
submultiplicative property / dimension bound.

So rank 18 is optimal! The lower bound is 18 (from the m*p dimension),
and we achieve it.

Let me construct the rank-18 decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120


def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return get_schoolbook()
    
    # Rank-18 decomposition using block structure
    # Two 3x3x3 schoolbook blocks sharing the same A
    rank = 18
    U = np.zeros((n * m, rank))  # (9, 18)
    V = np.zeros((m * p, rank))  # (18, 18)
    W = np.zeros((p * n, rank))  # (18, 18)
    
    # Block 1: k in [0, 3)
    # Term index s = 0..8, corresponding to (i,j,k) with k in [0,3)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                s = i * m + j * 3 + k  # 0..8
                U[i * m + j, s] = 1
                V[j * p + k, s] = 1
                W[k * n + i, s] = 1
    
    # Block 2: k in [3, 6)
    # Term index s = 9..17, corresponding to (i,j,k) with k in [3,6)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                s = 9 + i * m + j * 3 + (k - 3)  # 9..17
                U[i * m + j, s] = 1
                V[j * p + k, s] = 1
                W[k * n + i, s] = 1
    
    # Verify the decomposition
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return get_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
    
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"U: {U.shape}, V: {V.shape}, W: {W.shape}, rank={rank}")
    
    # Check tensor decomposition
    max_err = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                total = 0.0
                for r in range(rank):
                    total += U[i*3+j, r] * V[j*6+k, r] * W[k*3+i, r]
                max_err = max(max_err, abs(total - 1.0))
    print(f"Max error: {max_err}")
