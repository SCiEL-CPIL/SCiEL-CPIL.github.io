"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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
        for c in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

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

    def _strassen_2x2x2_decomposition():
        """
        Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = AB.
        Returns U, V, W lists of length 7, each containing a dict or list of values for the 4 indices.
        U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        V indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        W indices: (0,0), (0,1), (1,0), (1,1)
        """
        # Strassen terms:
        # M1 = (a+d)(e+h) -> U: a=1, d=1; V: e=1, h=1; W: c00=1, c11=1
        # M2 = d(f-e)    -> U: d=1; V: e=-1, f=1; W: c11=1
        # M3 = (c+d)e    -> U: c=1, d=1; V: e=1; W: c10=1
        # M4 = a(f-h)    -> U: a=1; V: f=1, h=-1; W: c01=1
        # M5 = (a+b)h    -> U: a=1, b=1; V: h=1; W: c01=1
        # M6 = (c-a)(e+f)-> U: a=-1, c=1; V: e=1, f=1; W: c10=1
        # M7 = (b-d)(g+h)-> U: b=1, d=-1; V: g=1, h=1; W: c11=1
        
        # We represent U, V, W as lists of 4 values for each of the 7 terms.
        # U values: [a, b, c, d]
        # V values: [e, f, g, h]
        # W values: [c00, c01, c10, c11]
        
        terms = []
        
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]))
        # M3
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M4
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M5
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        # M6
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]))
        # M7
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]))
        
        return terms

    def _build_strassen_4x4x9():
        """
        Build a decomposition for 4x4x9 using 2x2x2 Strassen blocks.
        A is 4x4, B is 4x9.
        Split A into 2x2 blocks: A11, A12, A21, A22.
        Split B into 2x2 blocks (first 8 columns) and 2x1 block (last column).
        B = [B11 B12 B13 B14 | B15]
            [B21 B22 B23 B24 | B25]
        C = A B
        C = [C11 C12 C13 C14 | C15]
            [C21 C22 C23 C24 | C25]
        
        C11 = A11 B11 + A12 B21
        C12 = A11 B12 + A12 B22
        C13 = A11 B13 + A12 B23
        C14 = A11 B14 + A12 B24
        C15 = A11 B15 + A12 B25  (B15, B25 are 2x1)
        
        C21 = A21 B11 + A22 B21
        C22 = A21 B12 + A22 B22
        C23 = A21 B13 + A22 B23
        C24 = A21 B14 + A22 B24
        C25 = A21 B15 + A22 B25
        
        For the 2x2x2 products (C11..C14, C21..C24), we use Strassen rank 7.
        For the 2x2x1 products (C15, C25), we use schoolbook rank 4? 
        2x2x1: C_i = sum_j A_ij B_j.
        U: 2x2, V: 2x1, W: 1x2.
        Rank 4?
        T = sum_{i,j} e_{i,j} otimes e_{j,0} otimes e_{0,i}.
        4 terms.
        
        Total rank = 8 * 7 (for 2x2x2) + 2 * 4 (for 2x2x1) = 56 + 8 = 64.
        Wait, 8 blocks of 2x2x2?
        C11, C12, C13, C14 are 2x2x2? No.
        C11 is 2x2. It is the result of A11(2x2) * B11(2x2) + A12(2x2) * B21(2x2).
        So it is a sum of TWO 2x2x2 tensors.
        So for each C_ij (2x2), we have 2 summands, each rank 7.
        Total 8 blocks * 2 summands * 7 rank = 112.
        Plus 2 blocks (C15, C25) * 2 summands * 4 rank = 16.
        Total 128.
        This is worse than 144? No, 128 < 144.
        
        Can we merge the summands?
        C11 = A11 B11 + A12 B21.
        This is a single 2x2x2x2 tensor? No.
        It is a linear combination.
        We just add the rank-1 terms.
        
        Let's implement this.
        """
        strassen_terms = _strassen_2x2x2_decomposition()
        
        # We will accumulate U, V, W
        # U size: 4*4 = 16 rows
        # V size: 4*9 = 36 rows
        # W size: 9*4 = 36 rows
        
        # We don't know the final rank yet, so we use lists of dicts or append to lists.
        # Let's use a list of tuples (u_vals, v_vals, w_vals) for each rank-1 term.
        # u_vals: dict mapping (i,j) to value
        # v_vals: dict mapping (j,k) to value
        # w_vals: dict mapping (k,i) to value
        
        terms = []
        
        # Helper to add a Strassen term for a specific 2x2 block product
        def add_strassen_term(A_block_idx, B_block_idx, C_block_idx, col_offset, is_2x2):
            # A_block_idx: (bi, bj) where i is row block, j is col block?
            # A is 4x4. Blocks:
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # B is 4x9. Blocks:
            # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4) [2x1]
            # 5: (1,0), 6: (1,1), 7: (1,2), 8: (1,3), 9: (1,4) [2x1]
            # C is 4x9. Blocks:
            # 0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (0,4) [2x1]
            # 5: (1,0), 6: (1,1), 7: (1,2), 8: (1,3), 9: (1,4) [2x1]
            
            # We need to map the 2x2 Strassen indices to the 4x4/4x9 indices.
            # A_block (bi, aj): rows 2*bi..2*bi+1, cols 2*aj..2*aj+1
            # B_block (bj, bk): rows 2*bj..2*bj+1, cols 2*bk..2*bk+1 (if 2x2)
            # C_block (ci, ck): rows 2*ci..2*ci+1, cols 2*ck..2*ck+1 (if 2x2)
            
            # A_block_idx is (bi, aj)
            # B_block_idx is (bj, bk)
            # C_block_idx is (ci, ck)
            
            bi, aj = A_block_idx
            bj, bk = B_block_idx
            ci, ck = C_block_idx
            
            # Verify consistency: bj must be aj (A col matches B row)
            # And ci must be bi (A row matches C row)
            # And ck must be bk (B col matches C col)
            if bj != aj or ci != bi or ck != bk:
                return # Should not happen
            
            for u_vals, v_vals, w_vals in strassen_terms:
                # u_vals: [a, b, c, d] for A
                # v_vals: [e, f, g, h] for B
                # w_vals: [c00, c01, c10, c11] for C
                
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Map A indices
                # (0,0) -> a, (0,1) -> b, (1,0) -> c, (1,1) -> d
                a_idx = (2*bi, 2*aj)
                b_idx = (2*bi, 2*aj+1)
                c_idx = (2*bi+1, 2*aj)
                d_idx = (2*bi+1, 2*aj+1)
                
                u_dict[a_idx] = u_vals[0]
                u_dict[b_idx] = u_vals[1]
                u_dict[c_idx] = u_vals[2]
                u_dict[d_idx] = u_vals[3]
                
                # Map B indices
                # (0,0) -> e, (0,1) -> f, (1,0) -> g, (1,1) -> h
                # B col offset: bk is the block column index.
                # If is_2x2, cols are 2*bk, 2*bk+1
                # If not 2x2 (2x1), cols is 2*bk (only one col)
                
                if is_2x2:
                    e_idx = (2*bj, 2*bk)
                    f_idx = (2*bj, 2*bk+1)
                    g_idx = (2*bj+1, 2*bk)
                    h_idx = (2*bj+1, 2*bk+1)
                    
                    v_dict[e_idx] = v_vals[0]
                    v_dict[f_idx] = v_vals[1]
                    v_dict[g_idx] = v_vals[2]
                    v_dict[h_idx] = v_vals[3]
                else:
                    # 2x1 block: only e and g exist (col 0 of the block)
                    # f and h are 0
                    e_idx = (2*bj, 2*bk)
                    g_idx = (2*bj+1, 2*bk)
                    
                    v_dict[e_idx] = v_vals[0]
                    # v_dict[f_idx] = v_vals[1] # 0
                    v_dict[g_idx] = v_vals[2]
                    # v_dict[h_idx] = v_vals[3] # 0
                    # But Strassen terms might have non-zero f, h.
                    # If B is 2x1, then f=h=0.
                    # So we can just ignore f, h in V.
                    # But W will have c01, c11 which correspond to col 1.
                    # If C is 2x1, then c01=c11=0.
                    # So we must ensure that if B is 2x1, we don't add terms that require B col 1.
                    # This means we can only use Strassen terms where v_vals[1] == 0 and v_vals[3] == 0?
                    # No, Strassen is for 2x2x2.
                    # For 2x2x1, we should use a different decomposition.
                    
                # Map C indices
                # (0,0) -> c00, (0,1) -> c01, (1,0) -> c10, (1,1) -> c11
                if is_2x2:
                    c00_idx = (2*ck, 2*ci) # W row is k*n+i. k=2*ck+kk, i=2*ci+ii
                    # W index: (k, i)
                    # c00: k=2*ck, i=2*ci
                    # c01: k=2*ck+1, i=2*ci
                    # c10: k=2*ck, i=2*ci+1
                    # c11: k=2*ck+1, i=2*ci+1
                    
                    w_dict[(2*ck, 2*ci)] = w_vals[0]
                    w_dict[(2*ck+1, 2*ci)] = w_vals[1]
                    w_dict[(2*ck, 2*ci+1)] = w_vals[2]
                    w_dict[(2*ck+1, 2*ci+1)] = w_vals[3]
                else:
                    # 2x1 C block: only c00, c10 exist
                    w_dict[(2*ck, 2*ci)] = w_vals[0]
                    # w_dict[(2*ck+1, 2*ci)] = w_vals[1] # 0
                    w_dict[(2*ck, 2*ci+1)] = w_vals[2]
                    # w_dict[(2*ck+1, 2*ci+1)] = w_vals[3] # 0
                    # Again, Strassen terms might have non-zero c01, c11.
                    # If C is 2x1, we must discard those.
                    # This implies Strassen is not directly applicable to 2x2x1.
                    
                if is_2x2:
                    terms.append((u_dict, v_dict, w_dict))
                else:
                    # For 2x2x1, we use schoolbook: 4 terms
                    # C_i = sum_j A_ij B_j
                    # T = sum_{i,j} e_{i,j} otimes e_{j,0} otimes e_{0,i}
                    # We add 4 rank-1 terms manually
                    for ii in range(2):
                        for jj in range(2):
                            u_d = {(2*bi+ii, 2*aj+jj): 1}
                            v_d = {(2*bj+jj, 2*bk): 1}
                            w_d = {(2*bk, 2*ci+ii): 1}
                            terms.append((u_d, v_d, w_d))
                    
        # Add terms for C11..C14 (2x2 blocks)
        # C11 = A11 B11 + A12 B21
        add_strassen_term((0,0), (0,0), (0,0), 0, True)
        add_strassen_term((0,1), (1,0), (0,0), 0, True)
        
        # C12 = A11 B12 + A12 B22
        add_strassen_term((0,0), (0,1), (0,1), 0, True)
        add_strassen_term((0,1), (1,1), (0,1), 0, True)
        
        # C13 = A11 B13 + A12 B23
        add_strassen_term((0,0), (0,2), (0,2), 0, True)
        add_strassen_term((0,1), (1,2), (0,2), 0, True)
        
        # C14 = A11 B14 + A12 B24
        add_strassen_term((0,0), (0,3), (0,3), 0, True)
        add_strassen_term((0,1), (1,3), (0,3), 0, True)
        
        # C15 = A11 B15 + A12 B25 (2x1 blocks)
        add_strassen_term((0,0), (0,4), (0,4), 0, False)
        add_strassen_term((0,1), (1,4), (0,4), 0, False)
        
        # C21 = A21 B11 + A22 B21
        add_strassen_term((1,0), (0,0), (1,0), 0, True)
        add_strassen_term((1,1), (1,0), (1,0), 0, True)
        
        # C22 = A21 B12 + A22 B22
        add_strassen_term((1,0), (0,1), (1,1), 0, True)
        add_strassen_term((1,1), (1,1), (1,1), 0, True)
        
        # C23 = A21 B13 + A22 B23
        add_strassen_term((1,0), (0,2), (1,2), 0, True)
        add_strassen_term((1,1), (1,2), (1,2), 0, True)
        
        # C24 = A21 B14 + A22 B24
        add_strassen_term((1,0), (0,3), (1,3), 0, True)
        add_strassen_term((1,1), (1,3), (1,3), 0, True)
        
        # C25 = A21 B15 + A22 B25 (2x1 blocks)
        add_strassen_term((1,0), (0,4), (1,4), 0, False)
        add_strassen_term((1,1), (1,4), (1,4), 0, False)
        
        # Now convert terms to U, V, W matrices
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        
        return {"U": U, "V": V, "W": W}

    # Start with guaranteed correct schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions if time permits
    if _time_left() > 5:
        candidate = _build_strassen_4x4x9()
        if _validate(candidate["U"], candidate["V"], candidate["W"]):
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank:
                best = candidate
                best_rank = cand_rank
    
    return best
# EVOLVE-BLOCK-END
