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
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # z2 = (x10+x11)*y00 -> Z10:1, Z11:-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # z3 = x00*(y01-y11) -> Z01:1, Z11:1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # z4 = x11*(y10-y00) -> Z00:1, Z10:1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # z5 = (x00+x01)*y11 -> Z00:-1, Z01:1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # z6 = (x10-x00)*(y00+y01) -> Z11:1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # z7 = (x01-x11)*(y10+y11) -> Z00:1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    U = np.zeros((N, rank), dtype=np.int64)
    V = np.zeros((N, rank), dtype=np.int64)
    W = np.zeros((N, rank), dtype=np.int64)
    
    p = 0
    for (x_blocks, y_blocks, c_blocks) in level1:
        # Pre-compute formA and formB as dense arrays for speed
        # formA[r,s] is a vector of length 16 (flat indices)
        formA = np.zeros((2, 2, 16), dtype=np.int64)
        formB = np.zeros((2, 2, 16), dtype=np.int64)
        
        for (br, bc, coeff) in x_blocks:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    formA[r, s, idx] += coeff
        
        for (br, bc, coeff) in y_blocks:
            for r in range(2):
                for s in range(2):
                    idx = bidx(br, bc, r, s)
                    formB[r, s, idx] += coeff
        
        for (x_entries, y_entries, z_entries) in level2:
            # Build U contribution
            u_vec = np.zeros(16, dtype=np.int64)
            v_vec = np.zeros(16, dtype=np.int64)
            w_vec = np.zeros(16, dtype=np.int64)
            
            for (er, es, ecoeff) in x_entries:
                u_vec += ecoeff * formA[er, es, :]
            
            for (er, es, ecoeff) in y_entries:
                v_vec += ecoeff * formB[er, es, :]
            
            # W contribution
            for (zr, zs, zcoeff) in z_entries:
                for (cb_r, cb_c, ccoeff) in c_blocks:
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zs
                    w_idx = col * n + row
                    w_vec[w_idx] += zcoeff * ccoeff
            
            U[:, p] = u_vec
            V[:, p] = v_vec
            W[:, p] = w_vec
            
            p += 1
    
    # Check deadline before final conversion
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
