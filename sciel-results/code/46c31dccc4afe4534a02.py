import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 3, 4, 6
    # Shapes: A is n x m, B is m x p, C is n x p
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)  # W[k*n + i, r] where C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Actually the convention is W[k*n+i, r]. Let's be careful.
    # C[i, k] = sum_{j, r} A[i, j] B[j, k] W_{k, i, r}
    # The tensor T is the matrix multiplication tensor.
    # T_{i, j, k} = 1 if ... no, T is the structure tensor.
    # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r
    # where T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here U has size n*m, V has size m*p, W has size p*n.
    # Indexing:
    # U index: i * m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j * p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k * n + i  (k in 0..p-1, i in 0..n-1)  <-- Note the order k*n+i

    def build_baseline():
        rank = 66
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1
            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1
            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1
            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1
            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1
            U[A10, col] += 1; U[A00, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1
            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)

        return U, V, W

    def verify(U, V, W):
        # Check if sum_r U[:, r] V[:, r] W[:, r] == T_target
        # T_target is the matrix multiplication tensor.
        # We can check this by computing the product for random integer matrices A, B and comparing to C = A @ B.
        # This is a probabilistic test but very reliable with a few trials.
        # Alternatively, we can check the tensor equality directly.
        # Direct check:
        # T[i, j, k] = 1 if i < n, j < m, k < p? No.
        # The tensor T for matrix multiplication M_{n, m, p} is defined such that
        # (A B)_{i, k} = sum_{j} A_{i, j} B_{j, k} = sum_{r} (sum_{j} U_{i, j, r} A_{i, j}) (sum_{k} V_{j, k, r} B_{j, k}) W_{k, i, r} ?
        # No, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
        # This means T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
        # Wait, the indices in the task are:
        # U: (n*m, rank) -> index (i, j) flattened.
        # V: (m*p, rank) -> index (j, k) flattened.
        # W: (p*n, rank) -> index (k, i) flattened.
        # So T_{(i,j), (j,k), (k,i)}? No.
        # The tensor is a 3-tensor T of shape (n*m, m*p, p*n).
        # T_{A_idx, B_idx, C_idx} = 1 if A_idx corresponds to (i, j), B_idx to (j, k), C_idx to (k, i) and they form a valid multiplication path?
        # Actually, for matrix multiplication, the tensor T is such that:
        # T_{i, j, k} = 1 if the indices are consistent?
        # Let's look at the structure.
        # C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # In the bilinear algorithm, we compute C using r products.
        # C_{i, k} = sum_r (sum_j U_{i, j, r} A_{i, j}) (sum_k' V_{j, k', r} B_{j, k'}) W_{k, i, r}
        # This implies that the tensor T has non-zero entries only when the indices match up.
        # Specifically, T_{(i,j), (j,k), (k,i)} = 1.
        # So T is a permutation-like tensor (a 3-way identity).
        # Let's verify this interpretation.
        # If T_{a, b, c} = 1 iff a=(i,j), b=(j,k), c=(k,i), then
        # sum_r U_{a, r} V_{b, r} W_{c, r} should equal T_{a, b, c}.
        # So for a=(i,j), b=(j,k), c=(k,i), the sum should be 1.
        # For any other combination, the sum should be 0.
        
        # Let's construct the target tensor T_target.
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_target[a_idx, b_idx, c_idx] = 1

        # Now check if sum_r U[:, r] V[:, r] W[:, r] == T_target
        # This is a sum of outer products.
        # We can flatten and check.
        # T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow for large rank, but rank is ~66, dims are 12, 24, 18.
        # 12*24*18 = 5184 elements. 66 iterations. Very fast.
        T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.array_equal(T_computed, T_target)

    U, V, W = build_baseline()
    
    if not verify(U, V, W):
        # Fallback: just return schoolbook? Or raise?
        # The baseline should be correct. If not, something is wrong.
        pass

    current_rank = U.shape[1]
    
    # Local search
    # We want to reduce the rank.
    # Strategy:
    # 1. Try to drop a column.
    #    If we drop column r, we have rank-1 columns.
    #    Does T_target lie in the span of the remaining rank-1 tensors?
    #    This is a linear algebra problem.
    #    Let T_1, ..., T_R be the rank-1 tensors.
    #    We want to know if T_target = sum_{r != d} c_r T_r for some c_r.
    #    This is equivalent to checking if T_target is in the column space of the matrix formed by flattening T_r (r != d).
    #    Since T_target is fixed, and T_r are fixed, we can check the rank of [T_1 ... T_R] vs [T_1 ... T_{d-1}, T_{d+1} ... T_R].
    #    Actually, we know T_target = sum_{r=1}^R 1 * T_r.
    #    If we drop T_d, we ask: is T_target in span(T_1, ..., T_{d-1}, T_{d+1}, ..., T_R)?
    #    This is true iff T_d is in span(T_1, ..., T_{d-1}, T_{d+1}, ..., T_R).
    #    So we just need to check if the set of tensors is linearly dependent, and if T_d is in the span of the others.
    
    def can_drop(U, V, W, d):
        R = U.shape[1]
        if R <= 1:
            return False
        # Flatten all tensors
        # T_r is a vector of length n*m*m*p*p*n
        # We want to check if T_d is in span(T_1, ..., T_{d-1}, T_{d+1}, ..., T_R)
        # This is equivalent to checking if the rank of {T_1, ..., T_R} is less than the rank of {T_1, ..., T_{d-1}, T_{d+1}, ..., T_R} + 1?
        # No, we know T_target = sum T_r.
        # If we remove T_d, the new sum is T_target - T_d.
        # We want to know if there exist coefficients c_r (r != d) such that sum_{r != d} c_r T_r = T_target.
        # This is a linear system: M x = b, where M is the matrix with columns T_r (r != d) and b is T_target.
        # We can solve this using least squares and check the residual.
        
        # To be efficient, we can use the fact that we only need to check if the residual is 0.
        # Let's construct the matrix M.
        # M has size (n*m*m*p*p*n) x (R-1).
        # This is huge: 12*24*18 = 5184 rows, 65 cols.
        # This is very small!
        
        dim = n*m * m*p * p*n
        M = np.zeros((dim, R - 1), dtype=np.int64)
        for i, r in enumerate(range(R)):
            if r == d:
                continue
            T_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            M[:, i] = T_r.flatten()
        
        b = np.einsum('i,j,k->ijk', U, V, W).flatten() # T_target
        
        # Solve M x = b
        # Use np.linalg.lstsq
        # Since entries are small integers, we can use exact arithmetic if we want, but float is likely fine for checking zero residual.
        # However, to be safe against floating point errors, we can check if the residual is close to 0.
        # Or better, use integer Gaussian elimination?
        # For now, let's try lstsq.
        
        try:
            x, residuals, rank, s = np.linalg.lstsq(M.astype(float), b.astype(float), rcond=None)
            # Check residual
            residual = np.linalg.norm(M.astype(float) @ x - b.astype(float))
            if residual < 1e-9:
                # It is in the span.
                # But we also need the new decomposition to be valid?
                # The task requires the decomposition to be exact.
                # If T_target is in the span, we can write T_target = sum_{r != d} c_r T_r.
                # This is a valid decomposition of rank R-1.
                # We can update U, V, W by scaling the columns.
                # But wait, the entries must be integer/half-integer.
                # The coefficients c_r might be rational.
                # If c_r are rational, we can scale the columns to make them integers.
                # Specifically, if c_r = p_r / q_r, we can multiply U[:, r], V[:, r], W[:, r] by p_r and divide by q_r?
                # No, we multiply the whole tensor T_r by c_r.
                # T_r = u_r \otimes v_r \otimes w_r.
                # c_r T_r = (c_r u_r) \otimes v_r \otimes w_r = u_r \otimes (c_r v_r) \otimes w_r = ...
                # So we can just multiply one of the vectors by c_r.
                # If c_r is rational, we can clear denominators.
                # But we also need to ensure the other entries remain integers.
                # If we multiply U[:, r] by c_r, and c_r is not integer, U[:, r] might become non-integer.
                # We can multiply all three vectors by the cube root of c_r? No.
                # We can multiply U[:, r] by c_r, V[:, r] by 1, W[:, r] by 1.
                # Then the entries of U[:, r] become c_r * (integers).
                # If c_r is a rational number p/q, then the entries are p/q * int.
                # To make them integers, we need q to divide the gcd of the entries?
                # Or we can multiply U[:, r] by p, V[:, r] by q, W[:, r] by 1? No, that changes the product.
                # We need u' \otimes v' \otimes w' = c (u \otimes v \otimes w).
                # If we set u' = c u, v' = v, w' = w, then u' might not be integer.
                # If we set u' = u, v' = c v, w' = w, same issue.
                # We can distribute the factor.
                # If c = p/q, we can set u' = p u, v' = q v, w' = w? No, product is pq (u \otimes v \otimes w).
                # We need product to be c.
                # Let u' = a u, v' = b v, w' = c w. Then abc = c.
                # We want a, b, c to be such that a u, b v, c w are integers.
                # Since u, v, w are integers, we need a, b, c to be rational.
                # We can choose a = c, b = 1, c' = 1.
                # Then a u might not be integer.
                # However, we can scale the entire column by a common factor later?
                # No, the columns are independent.
                # Actually, the problem allows "integer/half-integer entries".
                # So if c is a rational number, we can potentially find a, b, c such that abc = c and a u, b v, c w are half-integers.
                # This is a number theory problem.
                # For now, let's assume we can find such scaling.
                # Or, simpler: we can just check if the residual is 0, and if so, return True.
                # The actual construction of the new U, V, W is the hard part.
                # But if we can drop a column, the rank is reduced.
                # We can try to construct the new U, V, W.
                
                # Let's try to construct it.
                # We have T_target = sum_{r != d} c_r T_r.
                # We want U', V', W' such that sum_{r != d} U'[:, r] \otimes V'[:, r] \otimes W'[:, r] = T_target.
                # We can set U'[:, r] = U[:, r], V'[:, r] = V[:, r], W'[:, r] = c_r W[:, r].
                # Then we need c_r W[:, r] to be half-integers.
                # If c_r is integer, this is fine.
                # If c_r is half-integer, this is fine.
                # If c_r is other rational, we might need to scale U or V too.
                
                # For now, let's just return True to indicate a drop is possible.
                # The caller will handle the construction.
                return True
            else:
                return False
        except np.linalg.LinAlgError:
            return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        R = U.shape[1]
        for d in range(R):
            if time.time() > deadline:
                break
            if can_drop(U, V, W, d):
                # Drop column d
                # We need to update U, V, W.
                # We found that T_target is in span of the others.
                # Let's find the coefficients.
                dim = n*m * m*p * p*n
                M = np.zeros((dim, R - 1), dtype=np.int64)
                idx = 0
                for r in range(R):
                    if r == d:
                        continue
                    T_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                    M[:, idx] = T_r.flatten()
                    idx += 1
                b = np.einsum('i,j,k->ijk', U, V, W).flatten()
                
                x, residuals, rank, s = np.linalg.lstsq(M.astype(float), b.astype(float), rcond=None)
                
                # x contains the coefficients c_r for r != d
                # We need to scale the columns.
                # Let's try to scale W.
                # W'[:, r] = x[idx] * W[:, r]
                # Check if entries are half-integers.
                # If not, try scaling U or V.
                
                new_U = np.delete(U, d, axis=1)
                new_V = np.delete(V, d, axis=1)
                new_W = np.delete(W, d, axis=1)
                
                # Apply scaling
                # We want new_W[:, i] = x[i] * new_W[:, i] to be half-integers.
                # Let's check.
                scaled_W = new_W * x
                # Check if scaled_W are half-integers
                # Multiply by 2 and check if integers
                two_scaled_W = scaled_W * 2
                # If two_scaled_W are close to integers, we are good.
                if np.allclose(two_scaled_W, np.round(two_scaled_W), atol=1e-9):
                    # We can use these values.
                    # But we need to store them as integers or half-integers.
                    # The return type is not specified as integer array, but "integer/half-integer entries".
                    # We can return floats that are exactly half-integers.
                    U, V, W = new_U, new_V, scaled_W
                    improved = True
                    break
                else:
                    # Try scaling U
                    scaled_U = new_U * x
                    two_scaled_U = scaled_U * 2
                    if np.allclose(two_scaled_U, np.round(two_scaled_U), atol=1e-9):
                        U, V, W = scaled_U, new_V, new_W
                        improved = True
                        break
                    else:
                        # Try scaling V
                        scaled_V = new_V * x
                        two_scaled_V = scaled_V * 2
                        if np.allclose(two_scaled_V, np.round(two_scaled_V), atol=1e-9):
                            U, V, W = new_U, scaled_V, new_W
                            improved = True
                            break
                        else:
                            # Try splitting the factor
                            # This is more complex. For now, skip this drop.
                            pass

    # Final verification
    if not verify(U, V, W):
        # If something went wrong, return the baseline
        U, V, W = build_baseline()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
