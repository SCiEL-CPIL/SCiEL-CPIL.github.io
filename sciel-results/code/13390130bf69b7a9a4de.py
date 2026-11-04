import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Strategy: Use Strassen-like decomposition on 2x2 column blocks.
    # Group 10 columns into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each pair, solve 2x3 x 3x2 subproblem.
    #
    # For 2x3 x 3x2, we can use the following 11-product decomposition:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # Let B = [[b00, b01], [b10, b11], [b20, b21]]
    #
    # Define intermediate products:
    # M1 = (a00 + a10) * b00
    # M2 = (a00 + a10) * b01
    # M3 = (a01 + a11) * b10
    # M4 = (a01 + a11) * b11
    # M5 = (a02 + a12) * b20
    # M6 = (a02 + a12) * b21
    # M7 = (a00 - a10) * b00
    # M8 = (a00 - a10) * b01
    # M9 = (a01 - a11) * b10
    # M10 = (a01 - a11) * b11
    # M11 = (a02 - a12) * b20
    # M12 = (a02 - a12) * b21
    #
    # This gives 12 products, same as schoolbook. We need a better approach.
    #
    # Alternative: Use the fact that for 2x2 x 2x2, Strassen uses 7 products.
    # Split m=3 into m=2+1. For the m=2 part, use Strassen (7 products for 2x2x2x2).
    # For the m=1 part, use schoolbook (4 products for 2x1x1x2).
    # Total: 7 + 4 = 11 products per column pair.
    #
    # For the m=2 part (columns 0,1 of A, rows 0,1 of B):
    # A_sub = [[a00, a01], [a10, a11]] (2x2)
    # B_sub = [[b00, b01], [b10, b11]] (2x2)
    # C_sub = A_sub @ B_sub (2x2)
    #
    # Strassen for 2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C_sub[0,0] = M1 - M4 + M5 - M7
    # C_sub[0,1] = M3 + M4
    # C_sub[1,0] = M2 + M7
    # C_sub[1,1] = M1 - M2 + M3 + M6
    #
    # For the m=1 part (column 2 of A, row 2 of B):
    # A_sub2 = [[a02], [a12]] (2x1)
    # B_sub2 = [[b20, b21]] (1x2)
    # C_sub2 = A_sub2 @ B_sub2 (2x2)
    #
    # Schoolbook for 2x1 x 1x2:
    # M8 = a02 * b20
    # M9 = a02 * b21
    # M10 = a12 * b20
    # M11 = a12 * b21
    #
    # C_sub2[0,0] = M8
    # C_sub2[0,1] = M9
    # C_sub2[1,0] = M10
    # C_sub2[1,1] = M11
    #
    # Total C = C_sub + C_sub2:
    # C[0,0] = M1 - M4 + M5 - M7 + M8
    # C[0,1] = M3 + M4 + M9
    # C[1,0] = M2 + M7 + M10
    # C[1,1] = M1 - M2 + M3 + M6 + M11
    #
    # Total: 11 products per column pair, 5 pairs = 55 total rank.

    # Build the decomposition
    # For each column pair (k, k+1) where k in {0, 2, 4, 6, 8}:
    # We have 11 rank-1 terms.
    
    num_pairs = p // 2  # 5 pairs
    rank = num_pairs * 11  # 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen coefficients for 2x2 x 2x2:
    # M1 = (a00+a11)*(b00+b11): U has a00=1, a11=1; V has b00=1, b11=1
    # M2 = (a10+a11)*b00: U has a10=1, a11=1; V has b00=1
    # M3 = a00*(b10-b11): U has a00=1; V has b10=1, b11=-1
    # M4 = a11*(b01-b00): U has a11=1; V has b01=1, b00=-1
    # M5 = (a00+a01)*b11: U has a00=1, a01=1; V has b11=1
    # M6 = (a10-a00)*(b00+b01): U has a10=1, a00=-1; V has b00=1, b01=1
    # M7 = (a01-a11)*(b10+b11): U has a01=1, a11=-1; V has b10=1, b11=1
    
    # Strassen output coefficients:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Schoolbook for 2x1 x 1x2:
    # M8 = a02 * b20: U has a02=1; V has b20=1
    # M9 = a02 * b21: U has a02=1; V has b21=1
    # M10 = a12 * b20: U has a12=1; V has b20=1
    # M11 = a12 * b21: U has a12=1; V has b21=1
    
    # Output coefficients for schoolbook part:
    # C[0,0] += M8
    # C[0,1] += M9
    # C[1,0] += M10
    # C[1,1] += M11
    
    idx = 0
    for pair in range(num_pairs):
        k0 = pair * 2  # first column in pair
        k1 = pair * 2 + 1  # second column in pair
        
        # Strassen M1: (a00+a11)*(b00+b11)
        U[0*3+0, idx] = 1.0  # a00
        U[1*3+1, idx] = 1.0  # a11
        V[0*10+k0, idx] = 1.0  # b0k0
        V[1*10+k1, idx] = 1.0  # b1k1
        W[k0*2+0, idx] = 1.0  # C[0,k0]
        W[k1*2+1, idx] = 1.0  # C[1,k1]
        idx += 1
        
        # Strassen M2: (a10+a11)*b00
        U[1*3+0, idx] = 1.0  # a10
        U[1*3+1, idx] = 1.0  # a11
        V[0*10+k0, idx] = 1.0  # b0k0
        W[k1*2+1, idx] = -1.0  # C[1,k1] gets -M2
        idx += 1
        
        # Strassen M3: a00*(b10-b11)
        U[0*3+0, idx] = 1.0  # a00
        V[1*10+k0, idx] = 1.0  # b1k0
        V[1*10+k1, idx] = -1.0  # -b1k1
        W[k0*2+0, idx] = 0.0  # C[0,k0] no contribution
        W[k0*2+1, idx] = 1.0  # C[0,k1] gets M3
        W[k1*2+1, idx] = 1.0  # C[1,k1] gets M3
        idx += 1
        
        # Strassen M4: a11*(b01-b00)
        U[1*3+1, idx] = 1.0  # a11
        V[0*10+k1, idx] = 1.0  # b0k1
        V[0*10+k0, idx] = -1.0  # -b0k0
        W[k0*2+0, idx] = -1.0  # C[0,k0] gets -M4
        W[k0*2+1, idx] = 1.0  # C[0,k1] gets M4
        idx += 1
        
        # Strassen M5: (a00+a01)*b11
        U[0*3+0, idx] = 1.0  # a00
        U[0*3+1, idx] = 1.0  # a01
        V[1*10+k1, idx] = 1.0  # b1k1
        W[k0*2+0, idx] = 1.0  # C[0,k0] gets M5
        idx += 1
        
        # Strassen M6: (a10-a00)*(b00+b01)
        U[1*3+0, idx] = 1.0  # a10
        U[0*3+0, idx] = -1.0  # -a00
        V[0*10+k0, idx] = 1.0  # b0k0
        V[0*10+k1, idx] = 1.0  # b0k1
        W[k1*2+1, idx] = 1.0  # C[1,k1] gets M6
        idx += 1
        
        # Strassen M7: (a01-a11)*(b10+b11)
        U[0*3+1, idx] = 1.0  # a01
        U[1*3+1, idx] = -1.0  # -a11
        V[1*10+k0, idx] = 1.0  # b1k0
        V[1*10+k1, idx] = 1.0  # b1k1
        W[k0*2+0, idx] = -1.0  # C[0,k0] gets -M7
        W[k1*2+1, idx] = 1.0  # C[1,k1] gets M7
        idx += 1
        
        # Schoolbook M8: a02 * b20
        U[0*3+2, idx] = 1.0  # a02
        V[2*10+k0, idx] = 1.0  # b2k0
        W[k0*2+0, idx] = 1.0  # C[0,k0] gets M8
        idx += 1
        
        # Schoolbook M9: a02 * b21
        U[0*3+2, idx] = 1.0  # a02
        V[2*10+k1, idx] = 1.0  # b2k1
        W[k0*2+1, idx] = 1.0  # C[0,k1] gets M9
        idx += 1
        
        # Schoolbook M10: a12 * b20
        U[1*3+2, idx] = 1.0  # a12
        V[2*10+k0, idx] = 1.0  # b2k0
        W[k1*2+0, idx] = 1.0  # C[1,k0] gets M10
        idx += 1
        
        # Schoolbook M11: a12 * b21
        U[1*3+2, idx] = 1.0  # a12
        V[2*10+k1, idx] = 1.0  # b2k1
        W[k1*2+1, idx] = 1.0  # C[1,k1] gets M11
        idx += 1

    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = u_r.reshape(n, m)
        B_weighted = v_r.reshape(m, p)

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        C_decomp += w_r.reshape(p, n).T * prod_r

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))

    if error > 1e-10:
        # Fallback: return schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
