import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Build the matrix multiplication tensor T[i,j,k] = delta_{i*m+j, j*p+k, k*n+i}
    # T has shape (n*m, m*p, p*n) = (8, 32, 16)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Start with the Strassen-based decomposition of rank 56
    rank_56 = 56
    U56 = np.zeros((n*m, rank_56))
    V56 = np.zeros((m*p, rank_56))
    W56 = np.zeros((p*n, rank_56))

    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]

    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*m + a_col_off
        a01 = 0*m + a_col_off + 1
        a10 = 1*m + a_col_off
        a11 = 1*m + a_col_off + 1

        b00 = b_row_off*p + b_col_off
        b01 = b_row_off*p + b_col_off + 1
        b10 = (b_row_off+1)*p + b_col_off
        b11 = (b_row_off+1)*p + b_col_off + 1

        c00 = c_col_off*n + 0
        c01 = (c_col_off+1)*n + 0
        c10 = c_col_off*n + 1
        c11 = (c_col_off+1)*n + 1

        # P1 = (a00+a11)(b00+b11)
        U56[a00, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        W56[c11, col] += 1
        col += 1

        # P2 = (a10+a11)*b00
        U56[a10, col] += 1
        U56[a11, col] += 1
        V56[b00, col] += 1
        W56[c10, col] += 1
        W56[c11, col] += -1
        col += 1

        # P3 = a00*(b01-b11)
        U56[a00, col] += 1
        V56[b01, col] += 1
        V56[b11, col] += -1
        W56[c01, col] += 1
        W56[c11, col] += 1
        col += 1

        # P4 = a11*(b10-b00)
        U56[a11, col] += 1
        V56[b10, col] += 1
        V56[b00, col] += -1
        W56[c00, col] += 1
        W56[c10, col] += 1
        col += 1

        # P5 = (a00+a01)*b11
        U56[a00, col] += 1
        U56[a01, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += -1
        W56[c01, col] += 1
        col += 1

        # P6 = (a10-a00)*(b00+b01)
        U56[a10, col] += 1
        U56[a00, col] += -1
        V56[b00, col] += 1
        V56[b01, col] += 1
        W56[c11, col] += 1
        col += 1

        # P7 = (a01-a11)*(b10+b11)
        U56[a01, col] += 1
        U56[a11, col] += -1
        V56[b10, col] += 1
        V56[b11, col] += 1
        W56[c00, col] += 1
        col += 1

    assert col == rank_56

    # Verify the rank-56 decomposition is correct
    def verify(U, V, W, rank):
        # Check that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all (i,j,k)
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    val = 0.0
                    for r in range(rank):
                        val += U[i, r] * V[j, r] * W[k, r]
                    if abs(val - T[i, j, k]) > 1e-10:
                        return False
        return True

    assert verify(U56, V56, W56, rank_56), "Rank-56 decomposition failed verification"

    # Try to find a rank-55 decomposition by looking for a linear dependency
    # Strategy: fix U and V to the rank-56 values (dropping one column) and solve for W
    # This tests if the global output constraints admit a linear dependency
    
    # Try each of the 56 columns as the one to drop
    best_rank = rank_56
    best_U, best_V, best_W = U56, V56, W56

    for drop_col in range(rank_56):
        if time.time() > deadline - 10:
            break

        # Try dropping column drop_col
        # Keep all other columns
        keep_cols = [c for c in range(rank_56) if c != drop_col]
        new_rank = len(keep_cols)

        U_new = U56[:, keep_cols]
        V_new = V56[:, keep_cols]

        # We need to find W_new such that:
        # sum_{r in keep_cols} U_new[i,r] * V_new[j,r] * W_new[k,r] = T[i,j,k]
        
        # This is a linear system. Let's set it up.
        # For each (i,j,k), we have:
        # sum_r U_new[i,r] * V_new[j,r] * W_new[k,r] = T[i,j,k]
        
        # This is nonlinear in W because W_new[k,r] is multiplied by U_new[i,r]*V_new[j,r].
        # But if we fix U and V, it's linear in W.
        
        # Let's build the linear system:
        # For each (i,j,k): sum_r (U_new[i,r] * V_new[j,r]) * W_new[k,r] = T[i,j,k]
        
        # The unknowns are W_new[k,r] for k in range(p*n) and r in range(new_rank)
        # Total unknowns: p*n * new_rank = 16 * 55 = 880
        
        # The equations are: for each (i,j,k), one equation
        # Total equations: n*m * m*p * p*n = 8 * 32 * 16 = 4096
        
        # But many equations are redundant or zero on the RHS.
        # Let's only include equations where T[i,j,k] = 1.
        
        nonzero_indices = []
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    if T[i, j, k] == 1.0:
                        nonzero_indices.append((i, j, k))

        num_eqs = len(nonzero_indices)
        num_unknowns = p*n * new_rank

        # Build the coefficient matrix A and RHS vector b
        # A has shape (num_eqs, num_unknowns)
        # b has shape (num_eqs,)
        
        # This might be too large to build explicitly. Let's use a sparse approach.
        # Actually, 4096 equations and 880 unknowns is manageable.
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)  # All RHS are 1.0

        for eq_idx, (i, j, k) in enumerate(nonzero_indices):
            for r in range(new_rank):
                coeff = U_new[i, r] * V_new[j, r]
                if coeff != 0:
                    # Unknown index: k * new_rank + r
                    unk_idx = k * new_rank + r
                    A[eq_idx, unk_idx] = coeff

        # Solve the linear system
        try:
            # Use least squares since the system is overdetermined
            W_flat, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is valid
            W_new = W_flat.reshape(p*n, new_rank)
            
            # Verify the solution
            valid = True
            for i in range(n*m):
                if not valid:
                    break
                for j in range(m*p):
                    if not valid:
                        break
                    for k in range(p*n):
                        val = 0.0
                        for r in range(new_rank):
                            val += U_new[i, r] * V_new[j, r] * W_new[k, r]
                        expected = T[i, j, k]
                        if abs(val - expected) > 1e-8:
                            valid = False
                            break
            
            if valid:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                break  # Found a valid rank-55 decomposition

        except np.linalg.LinAlgError:
            continue

    # Return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
