import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Use rank 32 naive decomposition (safest baseline)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    #
    # Tensor entry T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] -> C[i,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    #
    # Rank = m = 4 * (n*p) = 4*8 = 32? No, naive rank is m.
    # Wait, naive algorithm does: for each j, C += A[:,j] * B[j,:]
    # This is m scalar multiplications of vectors?
    # Actually, the bilinear complexity of naive matrix mult is m * n * p / m = n*p? No.
    # Naive is:
    # for i in range(n):
    #   for k in range(p):
    #     C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The rank of the matrix multiplication tensor <n,m,p> is at most n*p * m? No.
    # The standard naive rank is m * (n*p) / something?
    # Actually, the trivial decomposition has rank n*p*m / min(n,p,m)? No.
    # The trivial decomposition is:
    # Each term r corresponds to a triple (i,j,k)? No, that's the tensor rank upper bound n*p*m.
    #
    # Let's use the standard "row-column" decomposition which has rank m * n * p / m = n*p?
    # No, the rank of the tensor for matrix multiplication A(n,m) x B(m,p) -> C(n,p)
    # is at most m * (number of non-zeros in C)?
    #
    # Actually, a very simple valid decomposition is:
    # For each j in 0..m-1:
    #   For each i in 0..n-1:
    #     For each k in 0..p-1:
    #       This doesn't reduce rank.
    #
    # The naive algorithm uses m "steps" (for j).
    # In each step, we compute A_col_i (vector) * B_row_k (vector) -> outer product added to C.
    # This is a rank 1 update for each (i,k) pair?
    #
    # Let's just construct the full rank n*p*m decomposition? No, that's 64.
    #
    # Better: The rank is at most m * min(n,p) * max(n,p)?
    #
    # Let's just use the identity:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We can view this as:
    # For each j, we have a rank-(n*p) tensor term? No.
    #
    # Actually, the rank of the matrix multiplication tensor is known to be m * n * p / m?
    #
    # Let's stick to the simplest correct one:
    # Rank = n * p * m? No, that's the number of entries in the sum, which is the "naive" count but not the tensor rank.
    #
    # Wait, the tensor is defined by T[i,j,k] = delta_{i,k}? No.
    # The multiplication tensor T has entries T[i,j,k] such that (A*B)[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] is 1 for all valid i,j,k? No.
    # T[i,j,k] is 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # So T[i,j,k] = 1 for all i,j,k in the ranges.
    #
    # The rank of this tensor (the "matrix multiplication tensor") is the bilinear complexity.
    #
    # A simple decomposition:
    # For each i, k:
    #   We want sum_j A[i,j] B[j,k].
    #   We can introduce a variable X_{i,j,k}? No.
    #
    # Let's use the "row" decomposition:
    # For each j:
    #   Let u_{i,j} = A[i,j]
    #   Let v_{j,k} = B[j,k]
    #   Then C[i,k] += u_{i,j} v_{j,k}
    #
    # This is a sum over j of outer products of vector u_j (size n) and vector v_j (size p).
    # Each outer product is a rank-1 matrix.
    # But we are decomposing the *tensor* into rank-1 tensors.
    # A rank-1 tensor is u_i v_j w_k.
    # The term A[i,j] B[j,k] can be written as u_{i,j} v_{j,k} w_{k,i}?
    # If we set:
    #   U[i*m+j, r] = 1 if r corresponds to (i,j,k)?
    #
    # Let's just use the rank-32 decomposition from the seed which was likely correct or close,
    # but the prompt says score 0.0.
    #
    # Let's generate a guaranteed correct rank-n*p*m decomposition (64) just to be safe,
    # or better, the rank-m * n * p decomposition?
    #
    # Actually, the rank is at most m * n * p.
    # But we can do better.
    #
    # Let's use the standard fact:
    # The rank of the <n,m,p> matrix multiplication tensor is at most m * n * p / min(n,m,p) ? No.
    #
    # Let's just output a correct rank-32 decomposition based on the "j" index being the rank dimension?
    # No, the rank dimension is the third dimension of U,V,W.
    #
    # Let's try rank = n*p = 8?
    # No.
    #
    # Let's try rank = m * n * p / m = n * p = 8?
    #
    # Actually, let's just do the naive expansion with rank = n * p * m = 64.
    # It is always correct.
    rank = n * p * m  # 2 * 4 * 4 = 32? No, 2*4*4=32.
    # Wait, n=2, m=4, p=4. n*p*m = 32.
    # Is the rank of the tensor 32?
    # The tensor has n*m*p = 32 entries.
    # If each entry is a separate term, the rank is 32.
    # Term r corresponds to (i,j,k).
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    #
    # Let's verify:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = sum_r delta_{r, (i,j,k)} * delta_{r, (i,j,k)} * delta_{r, (i,j,k)} ?
    #
    # If r = (i', j', k'), then:
    # U[i*m+j, r] is 1 only if i=i' and j=j'.
    # V[j*p+k, r] is 1 only if j=j' and k=k'.
    # W[k*n+i, r] is 1 only if k=k' and i=i'.
    #
    # So the product is 1 only if i=i', j=j', k=k'.
    # So the sum is 1 if (i,j,k) is a valid triple, else 0.
    # This is exactly the tensor T.
    #
    # So rank 32 is a valid decomposition.

    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
