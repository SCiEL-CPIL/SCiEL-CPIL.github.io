import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Helper to map local 2x2 block indices to global flat indices
    # block (br, bc) in a 4x4 matrix (2x2 blocks of size 2)
    # local (lr, lc) -> global row = br*2 + lr, global col = bc*2 + lc
    # A flat idx: (br*2+lr)*4 + (bc*2+lc)
    # B flat idx: (br*2+lr)*4 + (bc*2+lc)
    # W flat idx: (bc*2+lc)*4 + (br*2+lr)  <-- Note the swap for W
    
    def a_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def w_idx(br, bc, lr, lc):
        # C row = br*2+lr, C col = bc*2+lc
        # W index = col*4 + row
        return (bc*2 + lc)*4 + (br*2 + lr)

    def mk_dict(pairs):
        d = {}
        for k, v in pairs:
            d[k] = d.get(k, 0) + v
        return d

    # Standard Strassen 2x2 terms
    # (a, b, c)
    strassen_2x2_terms = [
        (mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)])),
        (mk_dict([((1,0),1), ((1,1),1)]), mk_dict([((0,0),1)]),              mk_dict([((1,0),1), ((1,1),-1)])),
        (mk_dict([((0,0),1)]),              mk_dict([((0,1),1), ((1,1),-1)]), mk_dict([((0,1),1), ((1,1),1)])),
        (mk_dict([((1,1),1)]),              mk_dict([((1,0),1), ((0,0),-1)]), mk_dict([((0,0),1), ((1,0),1)])),
        (mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)]),              mk_dict([((0,0),-1), ((0,1),1)])),
        (mk_dict([((1,0),1), ((0,0),-1)]),  mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)])),
        (mk_dict([((0,1),1), ((1,1),-1)]),  mk_dict([((1,0),1), ((1,1),1)]),   mk_dict([((0,0),1)])),
    ]
    
    # Outer Strassen for 4x4 (operating on 2x2 blocks)
    outer_strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]), # M2
        ([(0,0,1)],           [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]), # M3
        ([(1,1,1)],           [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # M4
        ([(0,0,1), (0,1,1)],  [(1,1,1)],           [(0,0,-1), (0,1,1)]), # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],  [(1,1,1)]),           # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],  [(0,0,1)]),           # M7
    ]
    
    col_idx = 0
    for outer_term in outer_strassen_terms:
        a_blocks, b_blocks, c_blocks = outer_term
        
        for inner_term in strassen_2x2_terms:
            a_comb, b_comb, c_comb = inner_term
            
            # Iterate over a_blocks to fill U
            for (br, bc, coeff) in a_blocks:
                for (lr, lc), a_val in a_comb.items():
                    if a_val == 0: continue
                    u_i = a_idx(br, bc, lr, lc)
                    U[u_i, col_idx] += coeff * a_val
            
            # Iterate over b_blocks to fill V
            for (br, bc, coeff) in b_blocks:
                for (lr, lc), b_val in b_comb.items():
                    if b_val == 0: continue
                    v_i = b_idx(br, bc, lr, lc)
                    V[v_i, col_idx] += coeff * b_val
            
            # Iterate over c_blocks and c_comb to fill W
            for (cbr, cbc, ccoeff_outer) in c_blocks:
                for (ir, ic), c_val_inner in c_comb.items():
                    if c_val_inner == 0: continue
                    w_i = w_idx(cbr, cbc, ir, ic)
                    W[w_i, col_idx] += ccoeff_outer * c_val_inner
            
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
