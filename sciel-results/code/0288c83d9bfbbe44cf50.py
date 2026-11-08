import numpy as np
import time

def construct():
    """
    Rank-49 tensor decomposition for 4x4 matrix multiplication via
    recursive Strassen (7^2 = 49 scalar products).
    
    U: (16, rank), V: (16, rank), W: (16, rank)
    Flat index: A[i,j] -> i*4+j
    W uses trace convention: W[k*n+i, r] where k=col, i=row
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n = 4
    N = 16
    rank = 49
    
    # U[row, col] corresponds to: coefficient of input element at flat index `row`
    # in the `col`-th product
    # Product p: U[*,p] * V[*,p] gives intermediate, then W[*,p] distributes to output
    
    # Recursive Strassen: 4x4 = 2x2 blocks of 2x2
    # Block (br,bc) of A covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # Flat index for A[2*br+r, 2*bc+s] = (2*br+r)*4 + (2*bc+s)
    
    def bidx(br, bc, r, s):
        return (2*br + r) * 4 + (2*bc + s)
    
    # Level-1: 7 block-level Strassen products for 2x2 blocks
    # Each product computes X * Y where X, Y are 2x2 matrices of blocks
    # X = sum of coeff * A_block, Y = sum of coeff * B_block
    # Result contributes to C blocks with coefficients
    level1 = [
        # M1 = (A00+A11)*(B00+B11) -> C00+=M, C11+=M
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)*B00 -> C10+=M, C11-=M
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00*(B01-B11) -> C01+=M, C11+=M
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11*(B10-B00) -> C00+=M, C10+=M
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)*B11 -> C00-=M, C01+=M
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)*(B00+B01) -> C11+=M
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)*(B10+B11) -> C00+=M
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Level-2: 7 scalar-level Strassen products for 2x2 matrices
    # Each product: x * y where x, y are linear combos of 2x2 entries
    # Result z contributes to output 2x2 entries
    level2 = [
        # z1 = (x00+x11)*(y00+y11) -> Z00:1, Z11:1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
        # z2 = (x10+x11)*y00 -> Z10:1, Z11:-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        # z3 = x00*(y01-y11) -> Z01:1, Z11:1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1, (1,1):1}),
        # z4 = x11*(y10-y00) -> Z00:1, Z10:1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1, (1,0):1}),
        # z5 = (x00+x01)*y11 -> Z00:-1, Z01:1
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        # z6 = (x10-x00)*(y00+y01) -> Z11:1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        # z7 = (x01-x11)*(y10+y11) -> Z00:1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]
    
    U = np.zeros((N, rank), dtype=np.int64)
    V = np.zeros((N, rank), dtype=np.int64)
    W = np.zeros((N, rank), dtype=np.int64)
    
    p = 0
    for (x_blocks, y_blocks, c_blocks) in level1:
        # Build the 2x2 "form" matrices: formA[r,s] is a dict mapping flat_idx -> coeff
        # for the X operand at position (r,s) within the 2x2 block structure
        # formA maps block position (br,bc) within the 2x2 to a 2x2 of flat indices
        
        # For each of the 4 positions (r,s) in the 2x2 block matrix:
        # formA[r,s] = {flat_idx: coeff} from x_blocks
        # formB[r,s] = {flat_idx: coeff} from y_blocks
        formA = {(r,s): {} for r in range(2) for s in range(2)}
        formB = {(r,s): {} for r in range(2) for s in range(2)}
        
        for (br, bc, coeff) in x_blocks:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    formA[(r,s)][idx] = formA[(r,s)].get(idx, 0) + coeff
        
        for (br, bc, coeff) in y_blocks:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    formB[(r,s)][idx] = formB[(r,s)].get(idx, 0) + coeff
        
        for (x_entries, y_entries, z_map) in level2:
            # Build U contribution: for each (er, es, ecoeff) in x_entries,
            # multiply by formA[er,es]
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (er, es, ecoeff) in x_entries:
                for flat_idx, val in formA[(er,es)].items():
                    u_dict[flat_idx] = u_dict.get(flat_idx, 0) + ecoeff * val
            
            for (er, es, ecoeff) in y_entries:
                for flat_idx, val in formB[(er,es)].items():
                    v_dict[flat_idx] = v_dict.get(flat_idx, 0) + ecoeff * val
            
            # W contribution: for each z position (zr, zs) with zcoeff in z_map,
            # and for each output block (cb_r, cb_c) with ccoeff in c_blocks:
            # output entry at (2*cb_r+zr, 2*cb_c+zs) gets zcoeff * ccoeff
            for (zr, zs), zcoeff in z_map.items():
                for (cb_r, cb_c, ccoeff) in c_blocks:
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zs
                    # Trace convention: W[k*n+i] where k=col, i=row
                    w_idx = col * n + row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + zcoeff * ccoeff
            
            for flat_idx, val in u_dict.items():
                if val != 0:
                    U[flat_idx, p] = val
            for flat_idx, val in v_dict.items():
                if val != 0:
                    V[flat_idx, p] = val
            for flat_idx, val in w_dict.items():
                if val != 0:
                    W[flat_idx, p] = val
            
            p += 1
    
    # Check deadline before final conversion
    if time.time() > deadline:
        # Return current best effort (already fully constructed)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
