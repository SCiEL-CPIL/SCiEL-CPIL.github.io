import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = 4, 4, 8  # 4x4 * 4x8 = 4x8
    
    # We'll use the rank-49 algorithm for 4x4x4 (based on Strassen recursion)
    # and apply it to each of the two 4x4x4 blocks (since 4x4x8 = two 4x4x4 problems)
    
    # The 4x4x4 rank-49 algorithm is a Strassen recursion:
    # Split 4x4 matrices into 2x2 blocks, use 7 Strassen products for each,
    # then split each 2x2 product into 2x2 sub-products.
    
    # Actually, the known rank-49 for 4x4x4 is:
    # 8 blocks of 2x2x2, each with 7 terms = 56... that's not 49.
    
    # Let me reconsider. The Bini-Capovani-Lotti-Romani (BCLR) algorithm gives rank 49 for 4x4x4.
    # But a simpler approach: use Strassen for the 4x4x4 case.
    
    # Strassen for 4x4x4:
    # Split into 2x2 blocks. We have 4 A-blocks, 4 B-blocks, 4 C-blocks.
    # Use 7 Strassen products on the 2x2x2 "super-matrix" level.
    # Each super-product is a 2x2x2 matrix multiplication, which needs 7 terms.
    # Total: 7 * 7 = 49 terms.
    
    # So the rank-49 algorithm for 4x4x4 is:
    # Level 1: 7 Strassen terms for 4x4x4 (treating each 2x2 block as a "scalar")
    # Level 2: Each of those 7 terms is a 2x2x2 multiplication, expanded to 7 terms
    # Total: 49 terms
    
    # Let's define the 49 terms for 4x4x4.
    
    # For 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4
    # Block structure (2x2 blocks of size 2x2):
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen at the 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now, each Mj is a 2x2x2 matrix multiplication.
    # We expand each Mj using Strassen for 2x2x2.
    
    # For a 2x2x2 multiplication X*Y=Z where X, Y, Z are 2x2:
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # Strassen:
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y10-y11)
    # S4 = x11(y01-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # z00 = S1 + S4 - S5 + S7
    # z01 = S3 + S5
    # z10 = S2 + S4
    # z11 = S1 - S2 + S3 + S6
    
    # Now, for the 4x4x4 case, each Mj involves sums/differences of 2x2 blocks.
    # We need to track the coefficients.
    
    # Let me define the terms as a list of (U_entries, V_entries, W_entries)
    # where each entry is (index, coefficient)
    
    # For 4x4x4, the flat indices are:
    # U: i*4+j where i is row of A, j is col of A (0-15)
    # V: j*4+k where j is row of B, k is col of B (0-15)
    # W: k*4+i where k is col of C, i is row of C (0-15)
    
    # Let me build the 49 terms programmatically.
    
    # First, define the 7 Strassen terms for 2x2x2 with local indices.
    # Local index for 2x2: (row, col) -> flat = row*2+col
    
    def get_strassen_2x2_terms():
        """Returns list of 7 tuples (u_list, v_list, w_list) for 2x2x2.
        Each list contains (local_flat_idx, coefficient).
        """
        terms = []
        # S1 = (x00+x11)(y00+y11)
        terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
        # S2 = (x10+x11)y00
        terms.append(([(2,1),(3,1)], [(0,1)], [(2,1)]))
        # S3 = x00(y10-y11)
        terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1)]))
        # S4 = x11(y01-y00)
        terms.append(([(3,1)], [(1,1),(0,-1)], [(1,1)]))
        # S5 = (x00+x01)y11
        terms.append(([(0,1),(1,1)], [(3,1)], [(1,1)]))
        # S6 = (x10-x00)(y00+y01)
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(2,1)]))
        # S7 = (x01-x11)(y10+y11)
        terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(3,1)]))
        return terms
    
    strassen_2x2 = get_strassen_2x2_terms()
    
    # Now, for the 4x4x4 Strassen recursion:
    # We need to define the 7 "meta-terms" at the 4x4 level.
    # Each meta-term j has:
    #   U_blocks: list of (block_idx, coeff) where block_idx is (ar, ac) for A
    #   V_blocks: list of (block_idx, coeff) where block_idx is (br, bc) for B
    #   W_blocks: list of (block_idx, coeff) where block_idx is (cr, cc) for C
    
    # Block indices for 4x4 (2x2 blocks of 2x2):
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen at 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    meta_terms = [
        # M1
        {
            'U': [((0,0),1), ((1,1),1)],
            'V': [((0,0),1), ((1,1),1)],
            'W': [((0,0),1), ((1,1),1)]
        },
        # M2
        {
            'U': [((1,0),1), ((1,1),1)],
            'V': [((0,0),1)],
            'W': [((1,0),1), ((1,1),-1)]
        },
        # M3
        {
            'U': [((0,0),1)],
            'V': [((1,0),1), ((1,1),-1)],
            'W': [((0,1),1), ((1,1),1)]
        },
        # M4
        {
            'U': [((1,1),1)],
            'V': [((0,1),1), ((0,0),-1)],
            'W': [((0,0),1), ((1,0),1)]
        },
        # M5
        {
            'U': [((0,0),1), ((0,1),1)],
            'V': [((1,1),1)],
            'W': [((0,1),1), ((0,0),-1)]
        },
        # M6
        {
            'U': [((1,0),1), ((0,0),-1)],
            'V': [((0,0),1), ((0,1),1)],
            'W': [((1,1),1)]
        },
        # M7
        {
            'U': [((0,1),1), ((1,1),-1)],
            'V': [((1,0),1), ((1,1),1)],
            'W': [((0,0),1)]
        },
    ]
    
    # For each meta-term, expand using Strassen 2x2.
    # Each meta-term j corresponds to a 2x2x2 multiplication where:
    #   The "X" matrix is formed from the U blocks
    #   The "Y" matrix is formed from the V blocks
    #   The "Z" matrix is formed from the W blocks
    
    # For each Strassen 2x2 term s, we combine with meta-term j.
    
    # Local block to global flat index:
    # For A (U): block (ar, ac) with local (lr, lc) -> global = (ar*2+lr)*4 + (ac*2+lc)
    # For B (V): block (br, bc) with local (lr, lc) -> global = (br*2+lr)*4 + (bc*2+lc)
    # For C (W): block (cr, cc) with local (lr, lc) -> global = (cc*2+lc)*4 + (cr*2+lr)
    
    def block_to_global_A(ar, ac, lr, lc):
        return (ar*2 + lr) * 4 + (ac*2 + lc)
    
    def block_to_global_B(br, bc, lr, lc):
        return (br*2 + lr) * 4 + (bc*2 + lc)
    
    def block_to_global_C(cr, cc, lr, lc):
        return (cc*2 + lc) * 4 + (cr*2 + lr)
    
    # Build the 49 terms for 4x4x4
    U_list_444 = []
    V_list_444 = []
    W_list_444 = []
    
    for meta in meta_terms:
        u_blocks = meta['U']  # list of ((ar,ac), coeff)
        v_blocks = meta['V']  # list of ((br,bc), coeff)
        w_blocks = meta['W']  # list of ((cr,cc), coeff)
        
        for (u_local, u_coeff) in strassen_2x2[0]:  # placeholder, we'll iterate properly
            pass
    
    # Let me redo this more carefully.
    # For each meta-term j, we have 7 Strassen 2x2 terms.
    # For each Strassen term s, we get one global term.
    
    # The Strassen 2x2 term s has:
    #   u_local: list of (local_flat_idx, coeff) for the 2x2 "X" matrix
    #   v_local: list of (local_flat_idx, coeff) for the 2x2 "Y" matrix
    #   w_local: list of (local_flat_idx, coeff) for the 2x2 "Z" matrix
    
    # The meta-term j has:
    #   u_blocks: list of ((ar,ac), coeff) - which 2x2 blocks of A contribute
    #   v_blocks: list of ((br,bc), coeff)
    #   w_blocks: list of ((cr,cc), coeff)
    
    # For a given Strassen term s and meta-term j:
    #   Global U entries: for each (local_idx, s_coeff) in u_local, for each ((ar,ac), m_coeff) in u_blocks:
    #       local_idx = lr*2+lc
    #       global_idx = (ar*2+lr)*4 + (ac*2+lc)
    #       entry = (global_idx, s_coeff * m_coeff)
    #   Similarly for V and W.
    
    for j, meta in enumerate(meta_terms):
        u_blocks = meta['U']
        v_blocks = meta['V']
        w_blocks = meta['W']
        
        for s in range(7):
            u_local = strassen_2x2[s][0]
            v_local = strassen_2x2[s][1]
            w_local = strassen_2x2[s][2]
            
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (local_idx, s_coeff) in u_local:
                lr = local_idx // 2
                lc = local_idx % 2
                for ((ar, ac), m_coeff) in u_blocks:
                    global_idx = block_to_global_A(ar, ac, lr, lc)
                    coeff = s_coeff * m_coeff
                    u_entries.append((global_idx, coeff))
            
            for (local_idx, s_coeff) in v_local:
                lr = local_idx // 2
                lc = local_idx % 2
                for ((br, bc), m_coeff) in v_blocks:
                    global_idx = block_to_global_B(br, bc, lr, lc)
                    coeff = s_coeff * m_coeff
                    v_entries.append((global_idx, coeff))
            
            for (local_idx, s_coeff) in w_local:
                lr = local_idx // 2
                lc = local_idx % 2
                for ((cr, cc), m_coeff) in w_blocks:
                    global_idx = block_to_global_C(cr, cc, lr, lc)
                    coeff = s_coeff * m_coeff
                    w_entries.append((global_idx, coeff))
            
            U_list_444.append(u_entries)
            V_list_444.append(v_entries)
            W_list_444.append(w_entries)
    
    # Now, for 4x4x8:
    # We have two independent 4x4x4 problems.
    # A is 4x4, B is 4x8, C is 4x8.
    # Split B into two 4x4 matrices: B_left (cols 0-3) and B_right (cols 4-7).
    # Split C into two 4x4 matrices: C_left (cols 0-3) and C_right (cols 4-7).
    
    # The first 49 terms handle B_left -> C_left.
    # The second 49 terms handle B_right -> C_right.
    
    U_list = []
    V_list = []
    W_list = []
    
    # First half: B cols 0-3, C cols 0-3
    for r in range(49):
        U_list.append(U_list_444[r])
        
        # For V: same as 4x4x4, but B is 4x4 with cols 0-3
        V_list.append(V_list_444[r])
        
        # For W: C cols 0-3
        W_list.append(W_list_444[r])
    
    # Second half: B cols 4-7, C cols 4-7
    # B_right is 4x4, but in the 4x8 matrix, its cols are 4-7.
    # So the flat index for B_right (j, k) where k is 0-3 is j*8 + (4+k).
    # C_right is 4x4, but in the 4x8 output, its cols are 4-7.
    # So the flat index for C_right (k, i) where k is 0-3 is (4+k)*4 + i.
    
    for r in range(49):
        # U is the same (A doesn't change)
        U_list.append(U_list_444[r])
        
        # For V: shift col index by 4
        v_entries_shifted = []
        for (idx, coeff) in V_list_444[r]:
            # idx = j*4 + k where k is 0-3
            j = idx // 4
            k = idx % 4
            new_idx = j * 8 + (4 + k)
            v_entries_shifted.append((new_idx, coeff))
        V_list.append(v_entries_shifted)
        
        # For W: shift col index by 4
        w_entries_shifted = []
        for (idx, coeff) in W_list_444[r]:
            # idx = k*4 + i where k is 0-3
            k = idx // 4
            i = idx % 4
            new_idx = (4 + k) * 4 + i
            w_entries_shifted.append((new_idx, coeff))
        W_list.append(w_entries_shifted)
    
    total_rank = len(U_list)
    
    # Verify
    def verify(U_terms, V_terms, W_terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for u_l, v_l, w_l in zip(U_terms, V_terms, W_terms):
                        u_val = sum(c for idx, c in u_l if idx == i*m+j)
                        v_val = sum(c for idx, c in v_l if idx == j*p+k)
                        w_val = sum(c for idx, c in w_l if idx == k*n+i)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U_list, V_list, W_list, n, m, p):
        # Fallback to naive
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list.append([(i*m+j, 1)])
                    V_list.append([(j*p+k, 1)])
                    W_list.append([(k*n+i, 1)])
        total_rank = len(U_list)
    
    rank = total_rank
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, coeff) in U_list[r]:
            U[idx, r] = coeff
        for (idx, coeff) in V_list[r]:
            V[idx, r] = coeff
        for (idx, coeff) in W_list[r]:
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
