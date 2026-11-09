import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define Strassen 2x2 x 2x4 decomposition
    # For X (2x2) at A rows ai, cols aj0:aj0+2
    # Y (2x4) at B rows bj0:bj0+2, cols bk0:bk0+4
    # Result contributes to C rows ai:ai+2, cols bk0:bk0+4
    
    # Strassen terms for X*Y where Y=[Y1|Y2], Y1,Y2 are 2x2
    # Each term: (coeff on X) * (coeff on Y) -> contributes to C entries
    
    # For X*Y1 (cols 0-1 of Y, i.e., global cols bk0, bk0+1):
    # M1: X=[1,0,0,1] Y=[1,0,0,1] -> C: [1,0,0,1] (C[0,0],C[1,1] of the 2x2 block)
    # M2: X=[0,0,1,1] Y=[1,0,0,0] -> C: [0,0,1,0]
    # M3: X=[1,0,0,0] Y=[0,1,-1,0] -> C: [0,1,0,1]
    # M4: X=[0,0,0,1] Y=[0,-1,1,0] -> C: [1,0,1,0]
    # M5: X=[1,1,0,0] Y=[0,0,0,1] -> C: [-1,1,0,0]
    # M6: X=[0,-1,1,0] Y=[1,1,0,0] -> C: [0,0,0,1]
    # M7: X=[0,1,0,-1] Y=[0,0,1,1] -> C: [1,0,0,0]
    
    # For X*Y2 (cols 2-3 of Y, i.e., global cols bk0+2, bk0+3):
    # M8: X=[1,0,0,1] Y=[1,0,0,1] -> C: [1,0,0,1]
    # M9: X=[0,0,1,1] Y=[1,0,0,0] -> C: [0,0,1,0]
    # M10: X=[1,0,0,0] Y=[0,1,-1,0] -> C: [0,1,0,1]
    # M11: X=[0,0,0,1] Y=[0,-1,1,0] -> C: [1,0,1,0]
    # M12: X=[1,1,0,0] Y=[0,0,0,1] -> C: [-1,1,0,0]
    # M13: X=[0,-1,1,0] Y=[1,1,0,0] -> C: [0,0,0,1]
    # M14: X=[0,1,0,-1] Y=[0,0,1,1] -> C: [1,0,0,0]
    
    # X is 2x2, indexed as [x00, x01, x10, x11] -> A[i, j] for i in {0,1}, j in {0,1}
    # Y is 2x4, indexed as [y00,y01,y02,y03, y10,y11,y12,y13] -> B[j, k] for j in {0,1}, k in {0,1,2,3}
    # C result is 2x4, indexed as [c00,c01,c02,c03, c10,c11,c12,c13] -> C[i, k] for i in {0,1}, k in {0,1,2,3}
    
    # W indexing: W[k*n+i] for C[i,k]
    # So C[i,k] -> W[k*2+i]
    
    # For the 2x2 x 2x4 block:
    # X entries: A[ai+0, aj0+0]=x00, A[ai+0, aj0+1]=x01, A[ai+1, aj0+0]=x10, A[ai+1, aj0+1]=x11
    # U index for A[i,j]: i*m + j = i*4 + j
    # Y entries: B[bj0+0, bk0+0]=y00, B[bj0+0, bk0+1]=y01, B[bj0+0, bk0+2]=y02, B[bj0+0, bk0+3]=y03
    #             B[bj0+1, bk0+0]=y10, B[bj0+1, bk0+1]=y11, B[bj0+1, bk0+2]=y12, B[bj0+1, bk0+3]=y13
    # V index for B[j,k]: j*p + k = j*8 + k
    # C entries: C[ai+0, bk0+0]=c00, C[ai+0, bk0+1]=c01, C[ai+0, bk0+2]=c02, C[ai+0, bk0+3]=c03
    #             C[ai+1, bk0+0]=c10, C[ai+1, bk0+1]=c11, C[ai+1, bk0+2]=c12, C[ai+1, bk0+3]=c13
    # W index for C[i,k]: k*n + i = k*2 + i
    
    def add_block(ai, aj0, bj0, bk0, base_col):
        """Add 14 rank-1 terms for 2x2 x 2x4 block."""
        # X indices in U: A[ai+i, aj0+j] -> (ai+i)*4 + (aj0+j)
        x_idx = [[(ai+0)*4 + (aj0+0), (ai+0)*4 + (aj0+1)],
                  [(ai+1)*4 + (aj0+0), (ai+1)*4 + (aj0+1)]]
        # Y indices in V: B[bj0+i, bk0+j] -> (bj0+i)*8 + (bk0+j)
        y_idx = [[(bj0+0)*8 + (bk0+0), (bj0+0)*8 + (bk0+1), (bj0+0)*8 + (bk0+2), (bj0+0)*8 + (bk0+3)],
                  [(bj0+1)*8 + (bk0+0), (bj0+1)*8 + (bk0+1), (bj0+1)*8 + (bk0+2), (bj0+1)*8 + (bk0+3)]]
        # C indices in W: C[ai+i, bk0+j] -> (bk0+j)*2 + (ai+i)
        c_idx = [[(bk0+0)*2 + (ai+0), (bk0+1)*2 + (ai+0), (bk0+2)*2 + (ai+0), (bk0+3)*2 + (ai+0)],
                  [(bk0+0)*2 + (ai+1), (bk0+1)*2 + (ai+1), (bk0+2)*2 + (ai+1), (bk0+3)*2 + (ai+1)]]
        
        # Strassen terms for X*Y1 (cols 0-1 of Y)
        # M1: X=[1,0,0,1] Y=[1,0,0,0] (only y00,y11 from Y1) C=[1,0,0,1] (c00,c11)
        # Wait, I need to be more careful. Y1 = [[y00,y01],[y10,y11]]
        # M1 = (x00+x11)*(y00+y11)
        # U: x00=1, x11=1 -> U[x_idx[0][0]]=1, U[x_idx[1][1]]=1
        # V: y00=1, y11=1 -> V[y_idx[0][0]]=1, V[y_idx[1][1]]=1
        # W: c00=1, c11=1 -> W[c_idx[0][0]]=1, W[c_idx[1][1]]=1
        
        terms = []
        # Y1 terms (cols 0,1 of Y -> C cols 0,1)
        # M1: (x00+x11)*(y00+y11) -> c00+=1, c11+=1
        terms.append(([x_idx[0][0], x_idx[1][1]], [y_idx[0][0], y_idx[1][1]], [c_idx[0][0], c_idx[1][1]], [1,1], [1,1], [1,1]))
        # M2: (x10+x11)*y00 -> c10+=1
        terms.append(([x_idx[1][0], x_idx[1][1]], [y_idx[0][0]], [c_idx[1][0]], [1,1], [1], [1]))
        # M3: x00*(y01-y11) -> c01+=1, c11+=1
        terms.append(([x_idx[0][0]], [y_idx[0][1], y_idx[1][1]], [c_idx[0][1], c_idx[1][1]], [1], [1,-1], [1,1]))
        # M4: x11*(y10-y00) -> c00+=1, c10+=1
        terms.append(([x_idx[1][1]], [y_idx[1][0], y_idx[0][0]], [c_idx[0][0], c_idx[1][0]], [1], [1,-1], [1,1]))
        # M5: (x00+x01)*y11 -> c00-=1, c01+=1
        terms.append(([x_idx[0][0], x_idx[0][1]], [y_idx[1][1]], [c_idx[0][0], c_idx[0][1]], [1,1], [1], [-1,1]))
        # M6: (x10-x00)*(y00+y01) -> c11+=1
        terms.append(([x_idx[1][0], x_idx[0][0]], [y_idx[0][0], y_idx[0][1]], [c_idx[1][1]], [1,-1], [1,1], [1]))
        # M7: (x01-x11)*(y10+y11) -> c00+=1
        terms.append(([x_idx[0][1], x_idx[1][1]], [y_idx[1][0], y_idx[1][1]], [c_idx[0][0]], [1,-1], [1,1], [1]))
        
        # Y2 terms (cols 2,3 of Y -> C cols 2,3)
        # Y2 = [[y02,y03],[y12,y13]]
        # M8: (x00+x11)*(y02+y13) -> c02+=1, c13+=1
        terms.append(([x_idx[0][0], x_idx[1][1]], [y_idx[0][2], y_idx[1][3]], [c_idx[0][2], c_idx[1][3]], [1,1], [1,1], [1,1]))
        # M9: (x10+x11)*y02 -> c12+=1
        terms.append(([x_idx[1][0], x_idx[1][1]], [y_idx[0][2]], [c_idx[1][2]], [1,1], [1], [1]))
        # M10: x00*(y03-y13) -> c03+=1, c13+=1
        terms.append(([x_idx[0][0]], [y_idx[0][3], y_idx[1][3]], [c_idx[0][3], c_idx[1][3]], [1], [1,-1], [1,1]))
        # M11: x11*(y12-y02) -> c02+=1, c12+=1
        terms.append(([x_idx[1][1]], [y_idx[1][2], y_idx[0][2]], [c_idx[0][2], c_idx[1][2]], [1], [1,-1], [1,1]))
        # M12: (x00+x01)*y13 -> c02-=1, c03+=1
        terms.append(([x_idx[0][0], x_idx[0][1]], [y_idx[1][3]], [c_idx[0][2], c_idx[0][3]], [1,1], [1], [-1,1]))
        # M13: (x10-x00)*(y02+y03) -> c13+=1
        terms.append(([x_idx[1][0], x_idx[0][0]], [y_idx[0][2], y_idx[0][3]], [c_idx[1][3]], [1,-1], [1,1], [1]))
        # M14: (x01-x11)*(y12+y13) -> c02+=1
        terms.append(([x_idx[0][1], x_idx[1][1]], [y_idx[1][2], y_idx[1][3]], [c_idx[0][2]], [1,-1], [1,1], [1]))
        
        for col_offset, (u_idxs, v_idxs, w_idxs, u_vals, v_vals, w_vals) in enumerate(terms):
            col = base_col + col_offset
            for idx, val in zip(u_idxs, u_vals):
                U[idx, col] += val
            for idx, val in zip(v_idxs, v_vals):
                V[idx, col] += val
            for idx, val in zip(w_idxs, w_vals):
                W[idx, col] += val
    
    # 4 blocks:
    # Block 1: A1 (ai=0, aj0=0) x B1a (bj0=0, bk0=0) -> C[:,0:4]
    add_block(0, 0, 0, 0, 0)
    # Block 2: A2 (ai=0, aj0=2) x B2a (bj0=2, bk0=0) -> C[:,0:4]
    add_block(0, 2, 2, 0, 14)
    # Block 3: A1 (ai=0, aj0=0) x B1b (bj0=0, bk0=4) -> C[:,4:8]
    add_block(0, 0, 0, 4, 28)
    # Block 4: A2 (ai=0, aj0=2) x B2b (bj0=2, bk0=4) -> C[:,4:8]
    add_block(0, 2, 2, 4, 42)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
