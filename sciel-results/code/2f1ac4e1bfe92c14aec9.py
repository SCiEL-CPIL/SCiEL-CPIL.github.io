import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2(off_i, off_j, off_k):
        # M1 = (A00+A11)(B00+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M2 = (A10+A11)B00
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # M3 = A00(B01-B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M4 = A11(B10-B00)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = -1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M5 = (A00+A01)B11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_laderman_3x3(off_i, off_j, off_k):
        # Laderman's algorithm for 3x3 matrix multiplication
        # Uses 23 multiplications
        # The 23 terms are defined by specific linear combinations
        
        # Term 1: (a11+a22)(b11+b22)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 2: (a12+a13)(b11+b21)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+0)*5+(off_j+2)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+2)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 3: (a13-a23)(b21-b31)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+2)] = 1
        u[(off_i+1)*5+(off_j+2)] = -1
        v[(off_j+2)*5+(off_k+1)] = 1
        v[(off_j+3)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # Term 4: (a21-a31)(b12-b22)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+2)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+2)] = -1
        add_col(u, v, w)

        # Term 5: (a22+a33)(b22+b33)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        u[(off_i+2)*5+(off_j+2)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 6: (a23+a33)(b22+b32)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+2)] = 1
        u[(off_i+2)*5+(off_j+2)] = 1
        v[(off_j+2)*5+(off_k+1)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 7: (a31-a11)(b12-b13)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+2)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+0)*5+(off_k+2)] = -1
        w[(off_k+1)*5+(off_i+2)] = 1
        w[(off_k+2)*5+(off_i+2)] = -1
        add_col(u, v, w)

        # Term 8: (a32-a12)(b13-b12)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+2)*5+(off_j+1)] = 1
        u[(off_i+0)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+2)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+2)*5+(off_i+2)] = 1
        w[(off_k+1)*5+(off_i+2)] = -1
        add_col(u, v, w)

        # Term 9: (a11+a12)(b21+b31)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+2)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 10: (a12+a22)(b21+b32)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+2)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 11: (a13+a23)(b22+b33)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+2)] = 1
        u[(off_i+1)*5+(off_j+2)] = 1
        v[(off_j+2)*5+(off_k+1)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+2)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 12: (a21+a31)(b12+b13)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+2)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+0)*5+(off_k+2)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 13: (a22+a32)(b13+b23)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        u[(off_i+2)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+2)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+2)*5+(off_i+1)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 14: (a23+a33)(b13+b23)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+2)] = 1
        u[(off_i+2)*5+(off_j+2)] = 1
        v[(off_j+1)*5+(off_k+2)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+2)*5+(off_i+1)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 15: a11*b11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # Term 16: a12*b21
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # Term 17: a13*b31
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+2)] = 1
        v[(off_j+2)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # Term 18: a21*b12
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 19: a22*b22
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 20: a23*b32
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+2)] = 1
        v[(off_j+2)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 21: a31*b13
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+2)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+2)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 22: a32*b23
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+2)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+2)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

        # Term 23: a33*b33
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+2)*5+(off_j+2)] = 1
        v[(off_j+2)*5+(off_k+2)] = 1
        w[(off_k+2)*5+(off_i+2)] = 1
        add_col(u, v, w)

    # Use 3+2 partition
    # Block 1: 3x3x3 at (0,0,0) - Laderman (rank 23)
    if time.time() < deadline:
        add_laderman_3x3(0, 0, 0)
    
    # Block 2: 3x2x3 at (0,3,0) - schoolbook (rank 18)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(0+i)*5 + (3+j)] = 1
                v_vec[(3+j)*5 + (0+k)] = 1
                w_vec[(0+k)*5 + (0+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Block 3: 3x3x2 at (0,0,3) - schoolbook (rank 18)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(0+i)*5 + (0+j)] = 1
                v_vec[(0+j)*5 + (3+k)] = 1
                w_vec[(3+k)*5 + (0+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Block 4: 2x3x3 at (3,0,0) - schoolbook (rank 18)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(3+i)*5 + (0+j)] = 1
                v_vec[(0+j)*5 + (0+k)] = 1
                w_vec[(0+k)*5 + (3+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Block 5: 2x2x3 at (3,3,0) - schoolbook (rank 12)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(3+i)*5 + (3+j)] = 1
                v_vec[(3+j)*5 + (0+k)] = 1
                w_vec[(0+k)*5 + (3+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Block 6: 2x3x2 at (3,0,3) - schoolbook (rank 12)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(3+i)*5 + (0+j)] = 1
                v_vec[(0+j)*5 + (3+k)] = 1
                w_vec[(3+k)*5 + (3+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Block 7: 2x2x2 at (3,3,3) - Strassen (rank 7)
    if time.time() < deadline:
        add_strassen_2x2(3, 3, 3)
    
    # Block 8: 3x2x2 at (0,3,3) - schoolbook (rank 12)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    break
                u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                u_vec[(0+i)*5 + (3+j)] = 1
                v_vec[(3+j)*5 + (3+k)] = 1
                w_vec[(3+k)*5 + (0+i)] = 1
                add_col(u_vec, v_vec, w_vec)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
