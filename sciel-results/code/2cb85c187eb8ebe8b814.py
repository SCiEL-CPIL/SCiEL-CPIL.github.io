"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Row-vector parallel decomposition:
    # Treat A (n x m) as n rows, each row is a 1x m vector.
    # For each row i, compute the product of row_i (1x m) with B (m x p) to get a 1x p row of C.
    # The sub-problem for each row is a 1x m x p tensor.
    # A rank-9 decomposition for 1x4x9: for each column k in B's 9 columns, we have a scalar-vector mult.
    # Actually, the 1xm x p tensor T[i,j,k] = 1 if i=0 (row index), and the multiplication is sum_j A[j]*B[j,k].
    # The trivial rank for 1xm x p is m*p? No, it's m*p if we use scalar multiplications, but we can do better?
    # Actually, 1xm * mxp -> 1xp. The tensor is sum_j A_j * (e_j ⊗ B[:,j]).
    # The rank of 1xm x p is m*p in general? No, it's just m*p scalars A_j * B_{j,k}.
    # Wait, the tensor for 1xm x p has m*p non-zero entries in the "trivial" sense? No.
    # The tensor T is in R^{1 x m x p}. T[0, j, k] = 1 for all j, k.
    # The rank of this tensor (as a 3-tensor) is m*p? No, it's the rank of the matrix when flattened?
    # Actually, the multiplication tensor for 1xm x p is: C_k = sum_j A_j B_{j,k}.
    # This can be written as sum_{j,k} A_j B_{j,k} C_k.
    # Each term (A_j, B_{j,k}, C_k) is a rank-1 tensor.
    # But we can group: for fixed k, C_k = sum_j A_j B_{j,k}.
    # This is an inner product of A with the k-th column of B.
    # The rank is m*p if we use the standard decomposition? No.
    # Actually, the tensor T[0,j,k]=1 has rank m (as a matrix of size 1 x mp? No).
    # Let me think: T is a 3-tensor of shape (1, m, p).
    # T_{0jk} = 1.
    # We want to write T = sum_r U_r ⊗ V_r ⊗ W_r.
    # U_r is in R^1, V_r in R^m, W_r in R^p.
    # T_{0jk} = sum_r U_{r,0} V_{r,j} W_{r,k} = 1.
    # This means sum_r U_{r,0} (V_r ⊗ W_r) = J (the all-ones matrix of size m x p).
    # The rank of the all-ones matrix J (m x p) is 1.
    # So we can write J = V_1 ⊗ W_1 where V_1 = ones(m), W_1 = ones(p).
    # Then T = U_1 ⊗ V_1 ⊗ W_1 with U_1 = [1].
    # So the rank of the 1xm x p tensor is 1!
    # Wait, that's not right for matrix multiplication.
    # The multiplication tensor for (1, m, p) is NOT T_{0jk}=1.
    # The multiplication is C_k = sum_j A_j B_{j,k}.
    # The tensor that defines this is M_{j,k,l} where we contract A and B to get C.
    # Actually, the standard way: C_k = sum_j A_j B_{j,k}.
    # This is a linear map from (A, B) to C.
    # The tensor is in R^{m x p x p}? No.
    # Let's use the standard definition: the tensor T in R^{(1*m) x (m*p) x (1*p)} is not right.
    # The dimensions are: A has 1*m = m entries, B has m*p entries, C has 1*p = p entries.
    # The tensor T in R^{m x (mp) x p} with T[a, (j,k), c] = 1 if a=j and c=k, else 0.
    # So T[j, (j',k'), c] = 1 if j=j' and c=k'.
    # This is exactly the "diagonal" tensor. Its rank is m*p.
    # Because each (j, k) pair requires a separate rank-1 term: A_j * B_{j,k} * C_k.
    # Wait, no. A_j * B_{j,k} * C_k: the C_k is the same for all j.
    # So for fixed k, sum_j A_j B_{j,k} C_k.
    # This is C_k * (sum_j A_j B_{j,k}).
    # The term is (A_j) * (B_{j,k}) * (C_k).
    # For a fixed k, we have m terms: sum_j (A_j) * (B_{j,k}) * (C_k).
    # We can factor C_k out: C_k * sum_j A_j B_{j,k}.
    # But in the tensor decomposition, each rank-1 term is a tri-linear form.
    # The term for (j,k) is: u_j * v_{j,k} * w_k where u_j = A_j, v_{j,k} = B_{j,k}, w_k = C_k.
    # These are m*p independent tri-linear forms? 
    # Actually, the forms are: for each (j,k), the form is A_j * B_{j,k} * C_k.
    # Are these linearly independent? Yes, because each involves a unique combination of A_j and B_{j,k} and C_k.
    # Wait, C_k is shared. So the forms are A_1 B_{1,k} C_k, A_2 B_{2,k} C_k, ..., A_m B_{m,k} C_k for each k.
    # For different k, the C_k is different, so they are independent.
    # For the same k, A_j B_{j,k} C_k: these share C_k but have different A_j and B_{j,k}.
    # The space spanned by {A_j B_{j,k} C_k} for j=1..m is m-dimensional for fixed k.
    # And for different k, they are independent.
    # So total rank is m * p.
    # So for 1x4x9, rank is 4*9 = 36.
    # And for 4x4x9, we have 4 such rows, so total rank 4 * 36 = 144.
    # This is the same as the trivial decomposition!
    # So the "row-vector parallel" approach gives the same rank.
    # But the directive says to implement it, so I will.

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
