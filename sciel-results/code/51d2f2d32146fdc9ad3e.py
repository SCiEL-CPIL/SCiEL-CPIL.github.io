"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 115.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    def _time_left():
        return _deadline - time.time()
    
    def _validate(U, V, W):
        """Validate tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        rank = len(U[0])
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        # Use numpy for fast validation
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        # Compute tensor contraction
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # We can do this by iterating over c or using einsum
        # T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        # Check if T is all 1s
        # For small rank, explicit loop is fine and avoids einsum overhead if needed, 
        # but numpy is faster.
        # Let's use einsum for speed.
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def _schoolbook():
        """Standard schoolbook decomposition, rank = n*m*p = 144"""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_2x2x2_terms():
        """
        Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = AB.
        U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        V indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        W indices: (0,0), (0,1), (1,0), (1,1) corresponding to C indices (k,i)
        """
        # Terms: (U_vals [a,b,c,d], V_vals [e,f,g,h], W_vals [c00,c01,c10,c11])
        terms = []
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # M1
        terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1])) # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])) # M3
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])) # M4
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0])) # M5
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0])) # M6
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1])) # M7
        return terms

    def _build_kronecker_decomposition():
        """
        Build decomposition using Kronecker product structure.
        A (4x4) = [ A11 A12 ; A21 A22 ] where Ai are 2x2.
        B (4x9) = [ B11 B12 B13 B14 B15 ; B21 B22 B23 B24 B25 ] where Bi are 2x2 except last two are 2x1.
        Actually, let's group B columns into blocks of 2:
        B = [ B^(1) B^(2) B^(3) B^(4) ] where B^(1..3) are 4x2 and B^(4) is 4x1.
        Wait, p=9. 9 = 2*4 + 1.
        So we have 4 blocks of width 2? No, 4*2 = 8. Plus 1.
        So 4 blocks of 2x2 (for B) and 1 block of 2x1?
        B is 4x9.
        Rows of B: 4. Split into 2 row blocks of size 2.
        Cols of B: 9. Split into 4 col blocks of size 2, and 1 col block of size 1.
        
        Let's define the tensor as a sum of rank-1 terms.
        The multiplication tensor T is the sum over all (i,j,k) of e_i \otimes e_j \otimes e_k.
        
        We can view A as a 2x2 block matrix of 2x2 blocks.
        A = [ A00 A01 ; A10 A11 ]
        
        B can be viewed as a 2x(4*2+1) matrix.
        Let's split B into 2 row blocks (size 2) and 5 col blocks (sizes 2,2,2,2,1).
        B = [ B00 B01 B02 B03 B04 ; B10 B11 B12 B13 B14 ]
        where B00..B03, B10..B13 are 2x2, and B04, B14 are 2x1.
        
        C = A B.
        C is 4x9.
        C = [ C00 C01 C02 C03 C04 ; C10 C11 C12 C13 C14 ]
        where C00..C03, C10..C13 are 2x2, and C04, C14 are 2x1.
        
        C00 = A00 B00 + A01 B10
        C01 = A00 B01 + A01 B11
        ...
        C04 = A00 B04 + A01 B14
        
        This is a sum of 10 block products (5 col blocks * 2 row blocks of C).
        Each block product C_ij = A_0j B_0i + A_1j B_1i (indices shifted).
        Actually, C_{r, c_block} = A_{r, 0} B_{0, c_block} + A_{r, 1} B_{1, c_block}.
        
        For c_block = 0..3 (2x2 blocks), we have a 2x2x2 tensor structure for the block multiplication?
        No, the block multiplication is:
        C_block (2x2) = A_left (2x2) * B_top (2x2) + A_right (2x2) * B_bottom (2x2).
        This is not a single matrix multiplication. It's a sum of two.
        However, we can treat the whole thing as a linear map.
        
        Let's use the Kronecker product of the 2x2x2 Strassen tensor and a "column selection" tensor.
        
        Alternative:
        Rank of T_{n,m,p} <= Rank(T_{n1,m1,p1}) * Rank(T_{n2,m2,p2}) if T is a Kronecker product.
        Here, A is 4x4. We can view it as 2x2 blocks of 2x2.
        B is 4x9.
        
        Let's try to decompose B into 4x2 and 4x1 parts.
        B = [ B2 | B1 ] where B2 is 4x8 and B1 is 4x1.
        
        T_{4,4,9} = T_{4,4,8} + T_{4,4,1} (in terms of output columns).
        T_{4,4,8} can be viewed as 4 independent T_{4,4,2} tensors?
        T_{4,4,2} = T_{2,2,2} \otimes T_{2,1,1}? No.
        
        Let's stick to the explicit sum of Strassen blocks.
        
        Terms for C_{r, c} where r is row block (0,1) and c is col block (0..4).
        C_{r, c} = A_{r, 0} B_{0, c} + A_{r, 1} B_{1, c}.
        
        For c=0..3 (2x2 blocks):
        C_{r, c} (2x2) = A_{r, 0} (2x2) B_{0, c} (2x2) + A_{r, 1} (2x2) B_{1, c} (2x2).
        
        We can compute this using 2 Strassen decompositions?
        Or can we combine them?
        Let M = [ A_{r,0} A_{r,1} ] (2x4) and N_c = [ B_{0,c} ; B_{1,c} ] (4x2).
        C_{r,c} = M N_c.
        This is a 2x4 * 4x2 multiplication.
        Rank of T_{2,4,2}?
        Schoolbook rank = 2*4*2 = 16.
        Can we do better?
        
        Let's just use the schoolbook rank for the 2x4 * 4x2 block to be safe, or use Strassen on 2x2 blocks.
        If we use Strassen for 2x2 * 2x2, we get rank 7 for each product.
        C_{r,c} = A_{r,0} B_{0,c} + A_{r,1} B_{1,c}.
        This is a sum of two 2x2x2 tensors.
        Rank <= 7 + 7 = 14.
        
        There are 2 row blocks (r=0,1) and 4 col blocks (c=0..3).
        Total 8 such blocks.
        Rank contribution: 8 * 14 = 112.
        
        For c=4 (2x1 block):
        C_{r, 4} (2x1) = A_{r, 0} (2x2) B_{0, 4} (2x1) + A_{r, 1} (2x2) B_{1, 4} (2x1).
        This is a 2x2 * 2x1 multiplication, summed over 2 terms.
        Rank of T_{2,2,1} is 4 (schoolbook).
        So 2 * 4 = 8 per block.
        2 row blocks.
        Rank contribution: 2 * 8 = 16.
        
        Total rank = 112 + 16 = 128.
        This is better than 144.
        
        Can we do better for the 2x4 * 4x2 case?
        T_{2,4,2}.
        A is 2x4, B is 4x2, C is 2x2.
        C = A B.
        This is equivalent to multiplying two 2x2 matrices?
        Let A = [ A1 A2 ] where A1, A2 are 2x2.
        B = [ B1 ; B2 ] where B1, B2 are 2x2.
        C = A1 B1 + A2 B2.
        This is the sum of two 2x2x2 tensors.
        Rank <= 7 + 7 = 14.
        Is there a better decomposition for T_{2,4,2}?
        It is isomorphic to T_{2,2,2} \oplus T_{2,2,2} in terms of rank?
        Actually, T_{2,4,2} is the tensor for multiplying a 2x4 matrix by a 4x2 matrix.
        This is the same as multiplying two 2x2 matrices? No.
        The dimensions are different.
        However, the rank of T_{2,4,2} is known to be 14?
        Let's check.
        T_{2,2,2} has rank 7.
        T_{2,4,2} can be seen as two independent T_{2,2,2} tensors?
        C_{ij} = sum_{k=1}^4 A_{ik} B_{kj}.
        Split k=1,2 and k=3,4.
        C = A_1 B_1 + A_2 B_2.
        Where A_1 is 2x2, B_1 is 2x2, etc.
        So yes, it is a direct sum of two T_{2,2,2} tensors.
        Rank = 7 + 7 = 14.
        
        So 128 is a solid bound.
        
        Let's implement the 128 rank decomposition.
        """
        strassen_terms = _strassen_2x2x2_terms()
        
        terms = [] # List of (u_dict, v_dict, w_dict)
        
        # Helper to add a Strassen term for 2x2 * 2x2
        def add_strassen_2x2x2(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
            """
            Adds rank-7 terms for A[A_row_off:A_row_off+2, A_col_off:A_col_off+2] * B[B_row_off:B_row_off+2, B_col_off:B_col_off+2]
            contributing to C[C_row_off:C_row_off+2, C_col_off:C_col_off+2]
            """
            for u_vals, v_vals, w_vals in strassen_terms:
                u_d = {}
                v_d = {}
                w_d = {}
                
                # A indices (i, j)
                # (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
                u_d[(A_row_off + 0, A_col_off + 0)] = u_vals[0]
                u_d[(A_row_off + 0, A_col_off + 1)] = u_vals[1]
                u_d[(A_row_off + 1, A_col_off + 0)] = u_vals[2]
                u_d[(A_row_off + 1, A_col_off + 1)] = u_vals[3]
                
                # B indices (j, k)
                # (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
                v_d[(B_row_off + 0, B_col_off + 0)] = v_vals[0]
                v_d[(B_row_off + 0, B_col_off + 1)] = v_vals[1]
                v_d[(B_row_off + 1, B_col_off + 0)] = v_vals[2]
                v_d[(B_row_off + 1, B_col_off + 1)] = v_vals[3]
                
                # C indices (k, i) -> W index is (k, i)
                # (0,0)->c00, (0,1)->c01, (1,0)->c10, (1,1)->c11
                w_d[(C_col_off + 0, C_row_off + 0)] = w_vals[0]
                w_d[(C_col_off + 1, C_row_off + 0)] = w_vals[1]
                w_d[(C_col_off + 0, C_row_off + 1)] = w_vals[2]
                w_d[(C_col_off + 1, C_row_off + 1)] = w_vals[3]
                
                terms.append((u_d, v_d, w_d))
        
        # Helper to add schoolbook terms for 2x2 * 2x1
        def add_schoolbook_2x2x1(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
            """
            Adds rank-4 terms for A[A_row_off:A_row_off+2, A_col_off:A_col_off+2] * B[B_row_off:B_row_off+2, B_col_off]
            contributing to C[C_row_off:C_row_off+2, C_col_off]
            """
            for ii in range(2):
                for jj in range(2):
                    u_d = {(A_row_off + ii, A_col_off + jj): 1}
                    v_d = {(B_row_off + jj, B_col_off): 1}
                    w_d = {(C_col_off, C_row_off + ii): 1}
                    terms.append((u_d, v_d, w_d))
        
        # Loop over C row blocks (r=0,1) and C col blocks (c=0..4)
        for r in range(2):
            for c in range(5):
                # C_block is at rows 2*r .. 2*r+1, cols 2*c .. 2*c+1 (if c<4) or 2*c (if c==4)
                C_row_off = 2 * r
                C_col_off = 2 * c
                
                # C = A_{r,0} B_{0,c} + A_{r,1} B_{1,c}
                
                # Term 1: A_{r,0} * B_{0,c}
                # A_{r,0} is at rows 2*r..2*r+1, cols 0..1
                # B_{0,c} is at rows 0..1, cols 2*c..2*c+1 (if c<4) or 2*c (if c==4)
                
                A_row_off_1 = 2 * r
                A_col_off_1 = 0
                B_row_off_1 = 0
                B_col_off_1 = 2 * c
                
                if c < 4:
                    add_strassen_2x2x2(A_row_off_1, A_col_off_1, B_row_off_1, B_col_off_1, C_row_off, C_col_off)
                else:
                    add_schoolbook_2x2x1(A_row_off_1, A_col_off_1, B_row_off_1, B_col_off_1, C_row_off, C_col_off)
                
                # Term 2: A_{r,1} * B_{1,c}
                # A_{r,1} is at rows 2*r..2*r+1, cols 2..3
                # B_{1,c} is at rows 2..3, cols 2*c..2*c+1 (if c<4) or 2*c (if c==4)
                
                A_row_off_2 = 2 * r
                A_col_off_2 = 2
                B_row_off_2 = 2
                B_col_off_2 = 2 * c
                
                if c < 4:
                    add_strassen_2x2x2(A_row_off_2, A_col_off_2, B_row_off_2, B_col_off_2, C_row_off, C_col_off)
                else:
                    add_schoolbook_2x2x1(A_row_off_2, A_col_off_2, B_row_off_2, B_col_off_2, C_row_off, C_col_off)
        
        # Convert terms to U, V, W matrices
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j][c_idx] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c_idx] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c_idx] = val
        
        return {"U": U, "V": V, "W": W}

    # Start with guaranteed correct schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions if time permits
    if _time_left() > 5:
        candidate = _build_kronecker_decomposition()
        if _validate(candidate["U"], candidate["V"], candidate["W"]):
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank:
                best = candidate
                best_rank = cand_rank
    
    return best
# EVOLVE-BLOCK-END
