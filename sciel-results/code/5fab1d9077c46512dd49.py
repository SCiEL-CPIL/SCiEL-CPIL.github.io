import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 8
    
    # We will construct the rank-98 decomposition using Strassen recursion.
    # 4x4x8 is two 4x4x4 blocks.
    # Each 4x4x4 uses 7 Strassen terms at the top level.
    # Each top-level term is a 2x2x2 multiplication, which uses 7 Strassen terms.
    # Total 49 per block, 98 total.
    
    U = np.zeros((n*m, 98), dtype=float)
    V = np.zeros((m*p, 98), dtype=float)
    W = np.zeros((p*n, 98), dtype=float)
    
    # Strassen 2x2x2 terms
    # U: (i,j), V: (j,k), W: (k,i)
    # Indices 0..3 for 2x2
    # Term 1: (00+11)(00+11) -> W 00, 11
    # Term 2: (10+11)00 -> W 10
    # Term 3: 00(10-11) -> W 01
    # Term 4: 11(01-00) -> W 01
    # Term 5: (00+01)11 -> W 01
    # Term 6: (10-00)(00+01) -> W 10
    # Term 7: (01-11)(10+11) -> W 11
    
    def get_2x2_strassen_terms():
        terms = []
        # u: list of (idx, coeff), v: list of (idx, coeff), w: list of (idx, coeff)
        # idx: 0=00, 1=01, 2=10, 3=11
        
        # S1
        terms.append(([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]))
        # S2
        terms.append(([(2,1), (3,1)], [(0,1)], [(2,1)]))
        # S3
        terms.append(([(0,1)], [(2,1), (3,-1)], [(1,1)]))
        # S4
        terms.append(([(3,1)], [(1,1), (0,-1)], [(1,1)]))
        # S5
        terms.append(([(0,1), (1,1)], [(3,1)], [(1,1)]))
        # S6
        terms.append(([(2,1), (0,-1)], [(0,1), (1,1)], [(2,1)]))
        # S7
        terms.append(([(1,1), (3,-1)], [(2,1), (3,1)], [(3,1)]))
        
        return terms

    strassen_2x2 = get_2x2_strassen_terms()
    
    # Strassen 4x4x4 terms (Meta terms)
    # A, B, C are 4x4, split into 2x2 blocks (00, 01, 10, 11)
    # Block indices: 0=00, 1=01, 2=10, 3=11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    meta_terms = [
        # M1
        {'U': [((0,0),1), ((1,1),1)], 'V': [((0,0),1), ((1,1),1)], 'W': [((0,0),1), ((1,1),1)]},
        # M2
        {'U': [((1,0),1), ((1,1),1)], 'V': [((0,0),1)], 'W': [((1,0),1), ((1,1),-1)]},
        # M3
        {'U': [((0,0),1)], 'V': [((1,0),1), ((1,1),-1)], 'W': [((0,1),1), ((1,1),1)]},
        # M4
        {'U': [((1,1),1)], 'V': [((0,1),1), ((0,0),-1)], 'W': [((0,0),1), ((1,0),1)]},
        # M5
        {'U': [((0,0),1), ((0,1),1)], 'V': [((1,1),1)], 'W': [((0,1),1), ((0,0),-1)]},
        # M6
        {'U': [((1,0),1), ((0,0),-1)], 'V': [((0,0),1), ((0,1),1)], 'W': [((1,1),1)]},
        # M7
        {'U': [((0,1),1), ((1,1),-1)], 'V': [((1,0),1), ((1,1),1)], 'W': [((0,0),1)]},
    ]
    
    def block_to_global_A(ar, ac, lr, lc):
        return (ar*2 + lr) * 4 + (ac*2 + lc)
    
    def block_to_global_B(br, bc, lr, lc):
        return (br*2 + lr) * 4 + (bc*2 + lc)
    
    def block_to_global_C(cr, cc, lr, lc):
        return (cc*2 + lc) * 4 + (cr*2 + lr)
    
    rank = 0
    for half in range(2):
        # half 0: cols 0-3
        # half 1: cols 4-7
        
        for j, meta in enumerate(meta_terms):
            u_blocks = meta['U']
            v_blocks = meta['V']
            w_blocks = meta['W']
            
            for s in range(7):
                u_local = strassen_2x2[s][0]
                v_local = strassen_2x2[s][1]
                w_local = strassen_2x2[s][2]
                
                # U entries
                for (local_idx, s_coeff) in u_local:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    for ((ar, ac), m_coeff) in u_blocks:
                        global_idx = block_to_global_A(ar, ac, lr, lc)
                        coeff = s_coeff * m_coeff
                        U[global_idx, rank] = coeff
                        
                # V entries
                for (local_idx, s_coeff) in v_local:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    for ((br, bc), m_coeff) in v_blocks:
                        global_idx = block_to_global_B(br, bc, lr, lc)
                        if half == 1:
                            # Shift col by 4
                            j_idx = global_idx // 4
                            k_idx = global_idx % 4
                            global_idx = j_idx * 8 + (4 + k_idx)
                        coeff = s_coeff * m_coeff
                        V[global_idx, rank] = coeff
                        
                # W entries
                for (local_idx, s_coeff) in w_local:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    for ((cr, cc), m_coeff) in w_blocks:
                        global_idx = block_to_global_C(cr, cc, lr, lc)
                        if half == 1:
                            # Shift col by 4
                            k_idx = global_idx // 4
                            i_idx = global_idx % 4
                            global_idx = (4 + k_idx) * 4 + i_idx
                        coeff = s_coeff * m_coeff
                        W[global_idx, rank] = coeff
                        
                rank += 1
                
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
