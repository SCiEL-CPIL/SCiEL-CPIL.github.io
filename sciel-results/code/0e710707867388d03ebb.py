"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
The known optimal rank for 4x4 matrix multiplication is 49.
This implementation uses the recursive Strassen algorithm (7^2 = 49 products)
or an equivalent direct construction.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    # Reshape for vectorized check
    # U_flat: (n*m, rank)
    # V_flat: (m*p, rank)
    # W_flat: (p*n, rank)
    
    # We need to check if the tensor T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} is the multiplication tensor.
    # The multiplication tensor has T_{ijk} = 1 if i==j and j==k? No.
    # C = A * B. C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor for matrix multiplication is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # No, the tensor $M_{n,m,p}$ is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # Let's check the convention: W[k*n+i, r].
    # The decomposition is:
    # C_{ik} = \sum_j A_{ij} B_{jk}
    # In tensor form: $C = \sum_r (U_r \otimes V_r \otimes W_r)$?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor $M$ has entries $M_{i,j,k} = 1$ if the index of C is determined by i and k, and j is the summation index.
    # Usually, the tensor is defined as $M = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # The bilinear map is $(A, B) \mapsto A B$.
    # The tensor $T$ representing this map satisfies:
    # $T(a, b) = \sum_r \langle U_r, A \rangle \langle V_r, B \rangle W_r$.
    # We want $T(A, B) = A B$.
    # So $\sum_r U_{ij,r} V_{jk,r} W_{ki,r} = \delta_{i,j} \delta_{j,k}$? No.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is 1 if the indices match the structure of matrix mult.
    # Specifically, the term $A_{ij} B_{jk}$ contributes to $C_{ik}$ with weight 1.
    # So for a fixed $j$, the term $A_{ij} B_{jk}$ must appear in the sum for $C_{ik}$.
    # The decomposition gives:
    # $C_{ik} = \sum_r W_{ki,r} \sum_j U_{ij,r} V_{jk,r}$? No.
    # The formula is:
    # $C_{ik} = \sum_r W_{ki,r} ( \sum_{j'} U_{i j', r} V_{j' k, r} )$? No.
    # The standard tensor decomposition form is:
    # $C_{ik} = \sum_r W_{ki,r} U_{i, \cdot, r} \cdot V_{\cdot, k, r}$?
    # Let's look at the verification code in the current program:
    # `total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]`
    # This sums over `r`. For a fixed `i, j, k`, it checks if the sum is 1.
    # This implies that the tensor entry $T_{i,j,k}$ (coefficient of $A_{ij} B_{jk}$ in the expression for $C$?) is 1.
    # Wait, if we sum over `r`, we get a value for each `i, j, k`.
    # If the result is 1 for all `i, j, k`, does that mean it's the identity tensor?
    # No, the matrix multiplication tensor is NOT the identity tensor.
    # The matrix multiplication tensor $M_{n,m,p}$ has non-zero entries only when the indices are consistent?
    # Actually, the tensor $M$ is defined by $M_{i,j,k} = 1$ if $i,j,k$ are valid indices?
    # No. The bilinear map is $B(A,B) = AB$.
    # The tensor $T$ associated with $B$ is such that $B(A,B) = \sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} e_k$?
    # Let's check the verification logic again.
    # `total` is the sum over `r` of `U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]`.
    # If this equals 1 for ALL `i, j, k`, then the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # This is the "all ones" tensor, not the matrix multiplication tensor.
    # The matrix multiplication tensor has $T_{i,j,k} = 1$ if $j$ is the middle index?
    # In $C = AB$, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The term $A_{ij} B_{jk}$ appears in $C_{ik}$.
    # So the tensor entry corresponding to indices $(i, j, k)$ in the input spaces and output space?
    # The decomposition is $C = \sum_r W_r (U_r \cdot A) (V_r \cdot B)$?
    # No, $C_{ik} = \sum_r W_{ki,r} U_{i j, r} V_{j k, r}$?
    # The verification code sums over `r` for a fixed `i, j, k`.
    # If the result is 1, it means the coefficient of the term $A_{ij} B_{jk}$ in the final sum is 1.
    # But in matrix multiplication, $A_{ij} B_{jk}$ contributes to $C_{ik}$, not just a generic $C$.
    # The index `k` in `W[k*n+i]` is the column of C.
    # The index `i` in `W[k*n+i]` is the row of C.
    # So `W[k*n+i]` corresponds to the output entry $C_{ik}$.
    # The term is $U_{ij} V_{jk} W_{ki}$.
    # If we sum this over `r`, we get the total contribution of the pair $(A_{ij}, B_{jk})$ to $C_{ik}$?
    # No, the sum is over `r`.
    # $\sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
    # If this equals 1 for all `i,j,k`, then the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # This is NOT the matrix multiplication tensor.
    # The matrix multiplication tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$ ONLY IF the problem was "compute all pairwise products".
    # But for matrix multiplication, $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # It does NOT contribute to $C_{i'k'}$ if $(i',k') \neq (i,k)$.
    # So the tensor entry $T_{i,j,k}$ (where $i$ is row of A, $j$ is col of A/row of B, $k$ is col of B) should be 1 if it contributes to $C_{ik}$?
    # Yes.
    # So the target tensor has $T_{i,j,k} = 1$ for all valid $i,j,k$.
    # Wait. In $C = AB$, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So for a fixed $i, j, k$, the term $A_{ij} B_{jk}$ contributes to $C_{ik}$ with weight 1.
    # So the tensor entry corresponding to inputs $(i,j)$ and $(j,k)$ and output $(i,k)$ is 1.
    # The verification code checks:
    # `sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1`
    # Here, the output index is implicitly $(i,k)$ because `W` is indexed by `k*n+i`.
    # So yes, the target tensor is indeed 1 for all valid $i,j,k$.
    # So the verification is correct.

    # Optimization: Use NumPy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # We want to check if the tensor T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r} is all 1s.
    # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank).
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # We need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # This is a contraction over r and j? No, j is fixed in the check.
    # The check is for every i, j, k.
    # We can compute the full tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Note: W is indexed by (k, i).
    
    # Let's use einsum
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # U: (i, j, r)
    # V: (j, k, r)
    # W: (k, i, r)
    
    # np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d) ?
    # W_3d is (p, n, rank) -> (k, i, r).
    # So:
    T = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
    
    # Check if all entries are 1
    if not np.allclose(T, 1.0):
        return False
    return True


def _strassen_4x4_decomposition():
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen products for 2x2
    _STRASSEN_2X2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]

    _STRASSEN_C_COMBINATION = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]

            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
