"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen: split 4x4 into 2x2 blocks, each block product
    uses 7 scalar multiplications (inner Strassen), giving 7*7 = 49 total.
    
    Convention: W[k*n + i, r] where n=4, so W index is (k*4+i) for the (i,k) element of C.
    U has shape (16, rank), V has shape (16, rank), W has shape (16, rank).
    U[i*4+j, r] corresponds to A[i,j]
    V[j*4+k, r] corresponds to B[j,k]
    W[k*4+i, r] corresponds to C[i,k]
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Inner Strassen for 2x2 matrices:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # 7 products:
    # P1 = (x00+x11)*(y00+y11)
    # P2 = (x10+x11)*y00
    # P3 = x00*(y01-y11)
    # P4 = x11*(y10-y00)
    # P5 = (x00+x01)*y11
    # P6 = (x10-x00)*(y00+y01)
    # P7 = (x01-x11)*(y10+y11)
    
    # Reconstruction:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For each product s, define the X elements (with signs) and Y elements (with signs)
    # X elements: list of (row, col, sign)
    # Y elements: list of (row, col, sign)
    inner_products = [
        # P1: (x00+x11)*(y00+y11)
        {
            'x': [(0, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1), (1, 1, 1)],
            'w': {0: 1, 3: 1}  # C11 += P1, C22 += P1
        },
        # P2: (x10+x11)*y00
        {
            'x': [(1, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1)],
            'w': {1: 1, 3: -1}  # C21 += P2, C22 -= P2
        },
        # P3: x00*(y01-y11)
        {
            'x': [(0, 0, 1)],
            'y': [(0, 1, 1), (1, 1, -1)],
            'w': {2: 1, 3: 1}  # C12 += P3, C22 += P3
        },
        # P4: x11*(y10-y00)
        {
            'x': [(1, 1, 1)],
            'y': [(1, 0, 1), (0, 0, -1)],
            'w': {0: 1, 1: 1}  # C11 += P4, C21 += P4
        },
        # P5: (x00+x01)*y11
        {
            'x': [(0, 0, 1), (0, 1, 1)],
            'y': [(1, 1, 1)],
            'w': {0: -1, 2: 1}  # C11 -= P5, C12 += P5
        },
        # P6: (x10-x00)*(y00+y01)
        {
            'x': [(1, 0, 1), (0, 0, -1)],
            'y': [(0, 0, 1), (0, 1, 1)],
            'w': {3: 1}  # C22 += P6
        },
        # P7: (x01-x11)*(y10+y11)
        {
            'x': [(0, 1, 1), (1, 1, -1)],
            'y': [(1, 0, 1), (1, 1, 1)],
            'w': {0: 1}  # C11 += P7
        },
    ]
    
    # Outer Strassen on 2x2 blocks:
    # A blocks: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B blocks: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    
    # 7 outer products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each outer product t, define A block elements (with signs) and B block elements (with signs)
    # Block: (br, bc, sign) where br, bc in {0,1}
    outer_products = [
        # M1: (A11+A22)(B11+B22)
        {
            'a': [((0, 0), 1), ((1, 1), 1)],
            'b': [((0, 0), 1), ((1, 1), 1)],
            'w': {0: 1, 3: 1}  # C11 += M1, C22 += M1
        },
        # M2: (A21+A22)*B11
        {
            'a': [((1, 0), 1), ((1, 1), 1)],
            'b': [((0, 0), 1)],
            'w': {1: 1, 3: -1}  # C21 += M2, C22 -= M2
        },
        # M3: A11*(B12-B22)
        {
            'a': [((0, 0), 1)],
            'b': [((0, 1), 1), ((1, 1), -1)],
            'w': {2: 1, 3: 1}  # C12 += M3, C22 += M3
        },
        # M4: A22*(B21-B11)
        {
            'a': [((1, 1), 1)],
            'b': [((1, 0), 1), ((0, 0), -1)],
            'w': {0: 1, 1: 1}  # C11 += M4, C21 += M4
        },
        # M5: (A11+A12)*B22
        {
            'a': [((0, 0), 1), ((0, 1), 1)],
            'b': [((1, 1), 1)],
            'w': {0: -1, 2: 1}  # C11 -= M5, C12 += M5
        },
        # M6: (A21-A11)*(B11+B12)
        {
            'a': [((1, 0), 1), ((0, 0), -1)],
            'b': [((0, 0), 1), ((0, 1), 1)],
            'w': {3: 1}  # C22 += M6
        },
        # M7: (A12-A22)*(B21+B22)
        {
            'a': [((0, 1), 1), ((1, 1), -1)],
            'b': [((1, 0), 1), ((1, 1), 1)],
            'w': {0: 1}  # C11 += M7
        },
    ]
    
    # For each (t, s), build one column of U, V, W
    for t_idx in range(7):
        outer = outer_products[t_idx]
        for s_idx in range(7):
            inner = inner_products[s_idx]
            col = t_idx * 7 + s_idx
            
            # Fill U: for each A block element and each inner X element
            for (ab, a_sign) in outer['a']:
                ab_row, ab_col = ab
                for (xr, xc, x_sign) in inner['x']:
                    # Global A index: (ab_row*2 + xr) * 4 + (ab_col*2 + xc)
                    gi = (ab_row * 2 + xr) * 4 + (ab_col * 2 + xc)
                    U[gi, col] += a_sign * x_sign
            
            # Fill V: for each B block element and each inner Y element
            for (bb, b_sign) in outer['b']:
                bb_row, bb_col = bb
                for (yr, yc, y_sign) in inner['y']:
                    # Global B index: (bb_row*2 + yr) * 4 + (bb_col*2 + yc)
                    gj = (bb_row * 2 + yr) * 4 + (bb_col * 2 + yc)
                    V[gj, col] += b_sign * y_sign
            
            # Fill W: for each outer W entry and each inner W entry
            for (oc, o_coeff) in outer['w'].items():
                # oc: 0=C11 block, 1=C21 block, 2=C12 block, 3=C22 block
                # Map to block (br, bc)
                block_map = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
                obr, obc = block_map[oc]
                for (ic, i_coeff) in inner['w'].items():
                    # ic: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    # Map to (r, c)
                    elem_map = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
                    er, ec = elem_map[ic]
                    # Global C element: (obr*2 + er, obc*2 + ec)
                    # C[i, k] where i = obr*2+er, k = obc*2+ec
                    # W index: k*4 + i = (obc*2+ec)*4 + (obr*2+er)
                    gk = (obc * 2 + ec) * 4 + (obr * 2 + er)
                    W[gk, col] += o_coeff * i_coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
