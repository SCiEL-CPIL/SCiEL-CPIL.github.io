            # r=0 (M1): (A00+A11)(B00+B11)
            # U: 0,3 -> 1
            # V: k0, 5+k0+1 -> 1
            # W: 2k0 (C00), 2k0+3 (C11) -> 1
            U[0][block_start+0] += 1
            U[3][block_start+0] += 1
            V[v_col_offset][block_start+0] += 1
            V[5 + v_col_offset + 1][block_start+0] += 1
            W[2*v_col_offset][block_start+0] += 1
            W[2*v_col_offset + 3][block_start+0] += 1
            
            # r=1 (M2): (A10+A11)B00
            # U: 2,3 -> 1
            # V: k0 -> 1
            # W: 2k0+1 (C10), 2k0+3 (C11) -> 1, -1
            U[2][block_start+1] += 1
            U[3][block_start+1] += 1
            V[v_col_offset][block_start+1] += 1
            W[2*v_col_offset + 1][block_start+1] += 1
            W[2*v_col_offset + 3][block_start+1] -= 1
            
            # r=2 (M3): A00(B01-B11)
            # U: 0 -> 1
            # V: k0+1, 5+k0+1 -> 1, -1
            # W: 2k0+2 (C01), 2k0+3 (C11) -> 1, 1
            U[0][block_start+2] += 1
            V[v_col_offset + 1][block_start+2] += 1
            V[5 + v_col_offset + 1][block_start+2] -= 1
            W[2*v_col_offset + 2][block_start+2] += 1
            W[2*v_col_offset + 3][block_start+2] += 1
            
            # r=3 (M4): A11(B10-B00)
            # U: 3 -> 1
            # V: 5+k0, k0 -> 1, -1
            # W: 2k0 (C00), 2k0+1 (C10) -> 1, 1
            U[3][block_start+3] += 1
            V[5 + v_col_offset][block_start+3] += 1
            V[v_col_offset][block_start+3] -= 1
            W[2*v_col_offset][block_start+3] += 1
            W[2*v_col_offset + 1][block_start+3] += 1
            
            # r=4 (M5): (A00+A10)B11
            # U: 0,2 -> 1
            # V: 5+k0+1 -> 1
            # W: 2k0 (C00), 2k0+2 (C01) -> -1, 1
            U[0][block_start+4] += 1
            U[2][block_start+4] += 1
            V[5 + v_col_offset + 1][block_start+4] += 1
            W[2*v_col_offset][block_start+4] -= 1
            W[2*v_col_offset + 2][block_start+4] += 1
            
            # r=5 (M6): (A01-A00)(B00+B10)
            # U: 1,0 -> 1, -1
            # V: k0, 5+k0 -> 1, 1
            # W: 2k0+3 (C11) -> -1
            U[1][block_start+5] += 1
            U[0][block_start+5] -= 1
            V[v_col_offset][block_start+5] += 1
            V[5 + v_col_offset][block_start+5] += 1
            W[2*v_col_offset + 3][block_start+5] -= 1
            
            # r=6 (M7): (A11-A10)(B01+B11)
            # U: 3,2 -> 1, -1
            # V: k0+1, 5+k0+1 -> 1, 1
            # W: 2k0 (C00) -> 1
            U[3][block_start+6] += 1
            U[2][block_start+6] -= 1
            V[v_col_offset + 1][block_start+6] += 1
            V[5 + v_col_offset + 1][block_start+6] += 1
            W[2*v_col_offset][block_start+6] += 1
