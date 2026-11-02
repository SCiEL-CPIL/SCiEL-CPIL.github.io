"""
Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.

The tensor T has indices (a, b, c) where:
  a = i*m + j  (i in [0,3), j in [0,3))  -> A[i,j]
  b = j*p + k  (j in [0,3), k in [0,6))  -> B[j,k]
  c = k*n + i  (k in [0,6), i in [0,3))  -> C[i,k]

T[a, b, c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c, else 0.

Strategy: Use structured decompositions based on:
1. Kronecker product structure: the tensor factors as (identity on i) ⊗ (multiplication tensor) ⊗ ...
2. For the 3x3x3 core, use Strassen-like rank-7 decomposition.
3. For the 3x3x6, exploit that p=6 = 3*2, so we can view it as three 3x3x2 blocks.

Actually, let's think more carefully. The tensor T[(i,j),(j',k),(k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}.

This is the "combinatorial" tensor. Its rank is at most n*m*p = 54 (schoolbook).

Key insight: This tensor is isomorphic to the matrix multiplication tensor for (n x m) * (m x p) -> (n x p) in the following sense:
- The standard matrix multiplication tensor M_{i,j,k} for (n x m)*(m x p) has M[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k]).

Wait, I need to be more careful. Let me re-derive.

The matrix multiplication operation: C = A * B where A is n×m, B is m×p, C is n×p.
C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]

The tensor that encodes this bilinear map is:
T ∈ R^{(n*m) × (m*p) × (n*p)}
T[(i,j), (j',k), (i'',k'')] = 1 if j=j' and i=i'' and k=k'', else 0.

This is what we have. The rank of this tensor is the bilinear complexity of matrix multiplication.

For 3×3 × 3×6:
- Schoolbook: 3*3*6 = 54
- We can use Strassen for the 3×3×3 part (rank 7 instead of 9) and handle the 6 outputs.

Actually, let's think of it as: C = A * B where A is 3×3, B is 3×6.
We can split B into two 3×3 matrices: B = [B1 | B2] where B1, B2 are 3×3.
Then C = [A*B1 | A*B2], and each A*B1, A*B2 is a 3×3×3 multiplication.

If we use Strassen for each 3×3×3 (rank 7), we get rank 14 total.
But wait, the A matrix is shared! In Strassen's algorithm, we compute 7 products of 2×2 blocks.
If we apply Strassen to A*B1 and A*B2 separately, we use the same A but different B blocks.

Actually, Strassen decomposes a 2×2 × 2×2 multiplication into 7 scalar multiplications.
For 3×3, we can use a 3×3 decomposition. The best known for 3×3×3 is rank 19 or 23.

Hmm, but the task says "rank-minimal exact tensor decomposition". Let me think about what's achievable.

For 3×3 × 3×6:
- The tensor is T[(i,j),(j',k),(k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
- This is equivalent to: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ e_i ⊗ e_j ⊗ e_k (in a 6-dimensional space)

Actually, a cleaner way: T can be seen as the tensor for the map (A, B) -> A*B where A is 3×3, B is 3×6.

Let me try a direct construction. The key observation is:
T[(i,j),(j',k),(k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}

This factors as: T = (⊗_i e_i ⊗ e_i) ⊗ (⊗_j e_j ⊗ e_j) ⊗ (⊗_k e_k ⊗ e_k)

Wait, that's not quite right in tensor product notation. Let me be explicit.

T is a 3-tensor with dimensions (n*m, m*p, p*n) = (9, 18, 18).

T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
T[a,b,c] = 1 if (i,i',j,j',k,k') satisfy i=i', j=j', k=k', else 0.

So T = sum_{i,j,k} E_{i,i'} ⊗ E_{j,j'} ⊗ E_{k,k'} where the indices are coupled.

More precisely: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} u_{i,j} ⊗ v_{j,k} ⊗ w_{k,i}
where u_{i,j}[a] = δ_{a, i*m+j}, v_{j,k}[b] = δ_{b, j*p+k}, w_{k,i}[c] = δ_{c, k*n+i}

This is the schoolbook decomposition with rank n*m*p = 54.

To get lower rank, we need to find a different set of vectors.

For 3×3×3, Strassen gives rank 7. For 3×3×6, we might be able to do better than 54.

Let me try: view the 3×6 output as two 3×3 outputs. Then C = A*B where B = [B1 | B2].
C1 = A*B1, C2 = A*B2.

If we use a rank-r decomposition for 3×3×3, then 3×3×6 has rank ≤ 2r.
With Strassen-like rank 7 for 3×3×3, we get rank 14.

But can we do better by sharing computation between C1 and C2?

Actually, the tensor structure is:
T[(i,j),(j',k),(k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}

For the "k" dimension (output columns), k ranges over 0..5. We can split k into k = 3*k2 + k1 where k2 ∈ {0,1}, k1 ∈ {0,1,2}.

Then T[(i,j),(j',3*k2+k1),(3*k2+k1,i')] = δ_{i,i'} δ_{j,j'} δ_{k2,k2'} δ_{k1,k1'}

This is a tensor product structure: T = T_3x3x3 ⊗ T_3x3x2 (roughly).

Hmm, this is getting complicated. Let me just try to construct an explicit low-rank decomposition.

Approach: Use the fact that for fixed i, the slice T[i, :, :] is a tensor in R^{(m*p) × (p*n)} = R^{18 × 18} that looks like:
T[i, (j',k), (k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}

For fixed i, this is: sum_{j,k} δ_{j,j'} δ_{k,k'} δ_{i,i'} = δ_{i,i'} * (identity-like structure in j,k)

Actually, for fixed i, T[i, (j',k), (k',i')] = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}

This is δ_{i,i'} times the tensor that maps (j',k) -> (k',j) with δ_{j,j'}δ_{k,k'}.

The tensor δ_{j,j'}δ_{k,k'} for j∈[0,3), k∈[0,6) is just the identity tensor in R^{(3*6) × (6*3)} = R^{18×18}, which has rank 18.

So for each i, the slice has "rank" 18 (in some sense), and there are 3 slices, giving total rank ≤ 3*18 = 54.

But we can do better. The full tensor is:
T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}

This is a sum of 54 rank-1 tensors. The question is whether they can be combined.

Let me try a different approach: use random optimization to find a low-rank decomposition.

Actually, let me try a structured approach. For 3×3×3, the best known rank is 19 (or 23 for a simpler construction). For 3×3×6, we can potentially do 2*19 = 38 or better.

Let me try to implement a search for low-rank decompositions using gradient descent or similar.

Given the time constraint, let me try:
1. Start with schoolbook rank 54
2. Try to find lower rank using random initialization + optimization
3. Use the structure of the tensor to guide the search

Actually, for a cleaner approach: let's use the fact that the tensor is:
T[a,b,c] = 1 if a%3 == c%3 (wait, no)

Let me re-index: a = i*3+j, b = j*6+k, c = k*3+i
T[a,b,c] = 1 iff (i,j,k) are consistent, i.e., i = a//3, j = a%3, j' = b//6, k = b%6, k' = c//3, i' = c%3, and i=i', j=j', k=k'.

So T[a,b,c] = 1 iff a//3 == c%3 AND a%3 == b//6 AND b%6 == c//3.

Let me verify: a = i*3+j, so a//3 = i, a%3 = j.
b = j*6+k, so b//6 = j, b%6 = k.
c = k*3+i, so c//3 = k, c%3 = i.

T[a,b,c] = 1 iff i == i' (a//3 == c%3) AND j == j' (a%3 == b//6) AND k == k' (b%6 == c//3).

OK so the tensor is: T[a,b,c] = 1 if (a//3 == c%3) and (a%3 == b//6) and (b%6 == c//3), else 0.

Now, let's try to find a rank-r decomposition. We need vectors U, V, W such that:
T[a,b,c] = sum_{r=0}^{R-1} U[a,r] * V[b,r] * W[c,r]

For the schoolbook, R=54 with U[a,r]=V[b,r]=W[c,r]=1 for the matching (i,j,k) triple.

Let me try to use a Kronecker product structure. The tensor T can be written as:
T = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)

= sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)

This is like a "cycle" tensor. 

For the 3×3×3 case, this is the standard matrix multiplication tensor, and Strassen gives rank 7.
For 3×3×6, we have an extra factor in the k dimension.

Let me try: T_{3x3x6} = T_{3x3x3} ⊗ I_2 (roughly). If T_{3x3x3} has rank r, then T_{3x3x6} has rank ≤ 2r.

With r=7 (Strassen), we get rank 14. But is this tight?

Actually, I think the correct relationship is:
T_{3x3x6}[(i,j),(j',k),(k',i')] = T_{3x3x3}[(i,j),(j',k1),(k1,i')] * δ_{k,k2} where k = 3*k2 + k1... no, that's not quite right.

Let me just try to construct an explicit decomposition.

For 3×3×3, Strassen's algorithm gives a rank-7 decomposition. Let me recall the 7 products:
For 2×2 matrices:
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B21-B22)
M4 = A22(B12-B11)
M5 = (A11+A12)B22
M6 = (A21-A11)(B11+B12)
M7 = (A12-A22)(B21+B22)

For 3×3, we can use a similar approach or use the known rank-19 or rank-23 decompositions.

Given the complexity, let me try a computational approach: use random vectors and optimize to minimize the error, starting from a moderate rank.

Here's my plan:
1. Define the target tensor T explicitly.
2. For a given rank R, initialize U, V, W randomly.
3. Use alternating least squares or gradient descent to minimize ||U ⊗ V ⊗ W - T||_F.
4. Check if the solution is exact (within tolerance).

Let me implement this.

"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Build the target tensor T of shape (n*m, m*p, p*n) = (9, 18, 18)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute T_reconstructed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum for efficiency
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_rec - T))
        return err < tol, err

    def schoolbook():
        """Return the schoolbook rank-54 decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower-rank decomposition
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    rng = np.random.RandomState(42)

    # Try ranks from low to high
    for target_rank in range(14, best_rank + 1):
        if time.time() > deadline - 30:
            break

        for attempt in range(10):
            if time.time() > deadline - 15:
                break

            U = rng.randn(n*m, target_rank)
            V = rng.randn(m*p, target_rank)
            W = rng.randn(p*n, target_rank)

            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)

            # Alternating least squares
            for it in range(50):
                if time.time() > deadline - 15:
                    break

                # Update U: for each a, solve min_U ||U[a,:] ⊙ (V⊙W) - T[a,:,:]||
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                # So U[a,:] = argmin ||sum_r u_r (V[:,r] ⊗ W[:,r]) - T[a,:,:]||
                # This is a linear system: U[a,:] * (V⊗W)^T = T[a,:,:].reshape(-1)

                # Build the design matrix: shape (m*p * p*n, target_rank)
                # D[:,r] = V[:,r] ⊗ W[:,r]
                # This is large: 18*18=324 rows, target_rank columns

                # For efficiency, do it in chunks
                for a in range(n*m):
                    # Design matrix for this a
                    D = np.zeros((m*p * p*n, target_rank))
                    for r in range(target_rank):
                        D[:, r] = (V[:, r] * W[:, r]).flatten()
                    # Solve: D @ U[a,:] = T[a,:,:].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(D, T[a, :].flatten(), rcond=None)[0]
                    except:
                        pass

                # Update V
                for b in range(m*p):
                    D = np.zeros((n*m * p*n, target_rank))
                    for r in range(target_rank):
                        D[:, r] = (U[:, r] * W[:, r]).flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(D, T[:, b, :].flatten(), rcond=None)[0]
                    except:
                        pass

                # Update W
                for c in range(p*n):
                    D = np.zeros((n*m * m*p, target_rank))
                    for r in range(target_rank):
                        D[:, r] = (U[:, r] * V[:, r]).flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(D, T[:, :, c].flatten(), rcond=None)[0]
                    except:
                        pass

            ok, err = verify(U, V, W, tol=1e-6)
            if ok:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break

        if best_rank == target_rank:
            break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
