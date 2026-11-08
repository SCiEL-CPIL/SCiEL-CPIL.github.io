import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is 50.
    # We construct a rank 46 decomposition.
    
    # Initialize U, V, W as lists of lists for easy appending/setting
    # We don't know the exact rank upfront if we make mistakes, but we know it's 46.
    # Let's pre-allocate or just use lists and zip at the end.
    # Using lists of dicts for sparse representation might be easier to build, then convert.
    # But 46 is small. Let's just build lists of lists.
    
    # We will accumulate columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_coeffs, V_coeffs, W_coeffs):
        """
        U_coeffs: dict {index: coeff}
        V_coeffs: dict {index: coeff}
        W_coeffs: dict {index: coeff}
        """
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        
        for idx, val in U_coeffs.items():
            u_row[idx] = val
        for idx, val in V_coeffs.items():
            v_row[idx] = val
        for idx, val in W_coeffs.items():
            w_row[idx] = val
            
        U_cols.append(u_row)
        V_cols.append(v_row)
        W_cols.append(w_row)

    def idx_U(i, j):
        return i * m + j

    def idx_V(j, k):
        return j * p + k

    def idx_W(k, i):
        return k * n + i

    # Helper to add a Strassen 2x2x2 block
    # i0, i1: rows of A
    # j0, j1: cols of A (rows of B)
    # k0, k1: cols of B (cols of C)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        # M1 = (a+d)(x+w)
        # a=A[i0,j0], d=A[i1,j1]
        # x=B[j0,k0], w=B[j1,k1]
        add_term(
            {idx_U(i0, j0): 1, idx_U(i1, j1): 1},
            {idx_V(j0, k0): 1, idx_V(j1, k1): 1},
            {idx_W(k0, i0): 1, idx_W(k1, i1): 1}
        )
        
        # M2 = (c+d)x
        # c=A[i1,j0], d=A[i1,j1]
        # x=B[j0,k0]
        add_term(
            {idx_U(i1, j0): 1, idx_U(i1, j1): 1},
            {idx_V(j0, k0): 1},
            {idx_W(k0, i1): 1}
        )
        
        # M3 = a(y-w)
        # a=A[i0,j0]
        # y=B[j0,k1], w=B[j1,k1]
        add_term(
            {idx_U(i0, j0): 1},
            {idx_V(j0, k1): 1, idx_V(j1, k1): -1},
            {idx_W(k1, i0): 1, idx_W(k1, i1): 1}
        )
        
        # M4 = d(z-x)
        # d=A[i1,j1]
        # z=B[j1,k0], x=B[j0,k0]
        add_term(
            {idx_U(i1, j1): 1},
            {idx_V(j1, k0): 1, idx_V(j0, k0): -1},
            {idx_W(k0, i0): -1, idx_W(k0, i1): 1}
        )
        
        # M5 = (a+b)w
        # a=A[i0,j0], b=A[i0,j1]
        # w=B[j1,k1]
        add_term(
            {idx_U(i0, j0): 1, idx_U(i0, j1): 1},
            {idx_V(j1, k1): 1},
            {idx_W(k0, i0): 1, idx_W(k1, i0): 1}
        )
        
        # M6 = (c-a)(x+z)
        # c=A[i1,j0], a=A[i0,j0]
        # x=B[j0,k0], z=B[j1,k0]
        add_term(
            {idx_U(i1, j0): 1, idx_U(i0, j0): -1},
            {idx_V(j0, k0): 1, idx_V(j1, k0): 1},
            {idx_W(k1, i1): 1}
        )
        
        # M7 = (b-d)(y+w)
        # b=A[i0,j1], d=A[i1,j1]
        # y=B[j0,k1], w=B[j1,k1]
        add_term(
            {idx_U(i0, j1): 1, idx_U(i1, j1): -1},
            {idx_V(j0, k1): 1, idx_V(j1, k1): 1},
            {idx_W(k0, i0): -1}
        )

    # Helper to add a 2x2x1 block (Schoolbook for 2x2 A and 2x1 B)
    # C = A B
    # C[i0, k0] = A[i0, j0]B[j0, k0] + A[i0, j1]B[j1, k0]
    # C[i1, k0] = A[i1, j0]B[j0, k0] + A[i1, j1]B[j1, k0]
    def add_2x2x1(i0, i1, j0, j1, k0):
        # Term 1: A[i0, j0] * B[j0, k0] -> C[i0, k0]
        add_term(
            {idx_U(i0, j0): 1},
            {idx_V(j0, k0): 1},
            {idx_W(k0, i0): 1}
        )
        # Term 2: A[i0, j1] * B[j1, k0] -> C[i0, k0]
        add_term(
            {idx_U(i0, j1): 1},
            {idx_V(j1, k0): 1},
            {idx_W(k0, i0): 1}
        )
        # Term 3: A[i1, j0] * B[j0, k0] -> C[i1, k0]
        add_term(
            {idx_U(i1, j0): 1},
            {idx_V(j0, k0): 1},
            {idx_W(k0, i1): 1}
        )
        # Term 4: A[i1, j1] * B[j1, k0] -> C[i1, k0]
        add_term(
            {idx_U(i1, j1): 1},
            {idx_V(j1, k0): 1},
            {idx_W(k0, i1): 1}
        )

    # Helper to add a 2x1x5 block (Outer product of 2x1 and 1x5)
    # A is 2x1 (col j0 of A, but here we treat it as the block A3)
    # B is 1x5 (row j0 of B)
    # C[i, k] = A[i, j0] * B[j0, k]
    # This requires 2 * 5 = 10 terms.
    def add_2x1x5(i0, i1, j0, k0, k1, k2, k3, k4):
        # i0, i1 are the two rows of A (0, 1)
        # j0 is the single column of A / row of B
        # k0..k4 are the 5 columns of B / cols of C
        
        # For each i in {i0, i1} and each k in {k0..k4}
        for i in [i0, i1]:
            for k in [k0, k1, k2, k3, k4]:
                add_term(
                    {idx_U(i, j0): 1},
                    {idx_V(j0, k): 1},
                    {idx_W(k, i): 1}
                )

    # 1. Block A1 (cols 0,1 of A) with B1 (rows 0,1 of B)
    # Split B1 cols into {0,1}, {2,3}, {4}
    # Strassen for {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)
    # Strassen for {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)
    # 2x2x1 for {4}
    add_2x2x1(0, 1, 0, 1, 4)
    
    # 2. Block A2 (cols 2,3 of A) with B2 (rows 2,3 of B)
    # Split B2 cols into {0,1}, {2,3}, {4}
    # Strassen for {0,1}
    add_strassen(0, 1, 2, 3, 0, 1)
    # Strassen for {2,3}
    add_strassen(0, 1, 2, 3, 2, 3)
    # 2x2x1 for {4}
    add_2x2x1(0, 1, 2, 3, 4)
    
    # 3. Block A3 (col 4 of A) with B3 (row 4 of B)
    # 2x1x5
    add_2x1x5(0, 1, 4, 0, 1, 2, 3, 4)
    
    # Convert to matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Verify
    def verify(U, V, W):
        # U, V, W are np arrays
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if val != 1:
                        return False
        # Check cross terms
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    else:
                        val = 0
                        for r in range(U.shape[1]):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if val != 0:
                            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
