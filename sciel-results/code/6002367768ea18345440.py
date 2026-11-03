import random
import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Rank is 28: four 2x2*2x2 Strassen products (7 mults each)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # X-index: 0=X00, 1=X01, 2=X10, 3=X11
    # Y-index: 0=Y00, 1=Y01, 2=Y10, 3=Y11
    # Z-index: 0=Z00, 1=Z01, 2=Z10, 3=Z11
    
    strassen_x = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    strassen_y = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]
    
    strassen_z = [
        {0: 1, 3: 1},      # M1 -> Z00, Z11
        {2: 1, 3: -1},     # M2 -> Z10, Z11
        {1: 1, 3: 1},      # M3 -> Z01, Z11
        {0: 1, 2: 1},      # M4 -> Z00, Z10
        {0: -1, 1: 1},     # M5 -> Z00, Z01
        {3: 1},            # M6 -> Z11
        {0: 1}             # M7 -> Z00
    ]
    
    # Four sub-products:
    # A is 2x4: A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # B is 4x4: B[0:2,0:2]=B11, B[0:2,2:4]=B12, B[2:4,0:2]=B21, B[2:4,2:4]=B22
    # C is 2x4: C[:,0:2] = A11*B11 + A12*B21, C[:,2:4] = A11*B12 + A12*B22
    #
    # Each sub-product: X is 2x2 block of A, Y is 2x2 block of B, Z is 2x2 block of C
    # (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    sub_products = [
        (0, 0, 0, 0, 0, 0),  # A11 * B11 -> C[:,0:2]
        (0, 2, 2, 0, 0, 0),  # A12 * B21 -> C[:,0:2]
        (0, 0, 0, 2, 0, 2),  # A11 * B12 -> C[:,2:4]
        (0, 2, 2, 2, 0, 2),  # A12 * B22 -> C[:,2:4]
    ]
    
    for sp_idx, (a_r0, a_c0, b_r0, b_c0, c_r0, c_c0) in enumerate(sub_products):
        for s in range(7):
            col = sp_idx * 7 + s
            
            # Fill U: A entries
            for x_idx, val in strassen_x[s].items():
                xi = x_idx // 2
                xj = x_idx % 2
                a_row = a_r0 + xi
                a_col = a_c0 + xj
                a_idx = a_row * m + a_col
                U[a_idx][col] += val
            
            # Fill V: B entries
            for y_idx, val in strassen_y[s].items():
                yi = y_idx // 2
                yj = y_idx % 2
                b_row = b_r0 + yi
                b_col = b_c0 + yj
                b_idx = b_row * p + b_col
                V[b_idx][col] += val
            
            # Fill W: C entries (W[k*n+i, r] convention where C[i,k])
            for z_idx, val in strassen_z[s].items():
                zi = z_idx // 2
                zj = z_idx % 2
                c_row = c_r0 + zi
                c_col = c_c0 + zj
                c_idx = c_col * n + c_row
                W[c_idx][col] += val
            
            # Deadline check
            if time.time() - start > 110:
                break
        if time.time() - start > 110:
            break
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
    
    # Verify with a random test case
    import random
    random.seed(42)
    n, m, p = DIMENSIONS
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B directly
    C_direct = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Compute via decomposition: C[i][k] = sum_r W[k*n+i][r] * (sum_j U[j*m+i]... 
    # Actually: M_r = dot(U_col_r, A_flat) * dot(V_col_r, B_flat)
    # C[i][k] = sum_r W[k*n+i][r] * M_r
    # Wait, the standard form is:
    # C[i,k] = sum_r U[i,r] * M_r * V[k,r]  -- NO
    # The tensor decomposition: T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # where T_{i,j,k} = delta_{i, j} * A_{i,k} ... no.
    # For matrix mult: C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # Tensor form: T_{i,j,k} where the multiplication tensor has T_{i,j,k} = 1 if j connects i and k
    # Decomposition: T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # So C_{i,k} = sum_r (sum_j U_{i,r} * A_{i,j}... 
    # Actually: C_{i,k} = sum_r W_{k,r} * (sum_j U_{i,r} * V_{j,r} * A_{i,j} * B_{j,k})
    # No, the correct form is:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} * A_{i,j}) * (sum_l V_{j,l,r} * B_{j,l}) * W_{k,r}
    # Hmm, let me just verify numerically.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # A_flat[i*m+j] = A[i,j], B_flat[j*p+k] = B[j,k], C_flat[k*n+i] = C[i,k]
    
    A_flat = [A[i][j] for i in range(n) for j in range(m)]
    B_flat = [B[j][k] for j in range(m) for k in range(p)]
    
    # M_r = sum_{a_idx} U[a_idx][r] * A_flat[a_idx] * sum_{b_idx} V[b_idx][r] * B_flat[b_idx]
    # C_flat[c_idx] = sum_r W[c_idx][r] * M_r
    
    M_vals = []
    for r in range(rank):
        u_dot = sum(U[a][r] * A_flat[a] for a in range(n * m))
        v_dot = sum(V[b][r] * B_flat[b] for b in range(m * p))
        M_vals.append(u_dot * v_dot)
    
    C_flat = [sum(W[c][r] * M_vals[r] for r in range(rank)) for c in range(p * n)]
    
    # Check
    max_err = 0
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            err = abs(C_flat[c_idx] - C_direct[i][k])
            max_err = max(max_err, err)
    
    print(f"Max error: {max_err}")
    assert max_err == 0, f"Decomposition incorrect! Max error: {max_err}"
    print("Verification passed!")
