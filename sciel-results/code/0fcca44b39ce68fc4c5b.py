        # 5. M5 = (x21+x32)(y11+y22)
        U[1*4 + j_offset, col] = 1
        U[2*4 + j_offset+1, col] = 1
        V[j_offset*4 + k_offset, col] = 1
        V[(j_offset+1)*4 + k_offset+1, col] = 1
        W[k_offset*3 + 1, col] = 1   # Z11
        W[(k_offset+1)*3 + 2, col] = 1 # Z22
        col += 1
        
        # 6. M6 = (x31+x32)y11
        U[2*4 + j_offset, col] = 1
        U[2*4 + j_offset+1, col] = 1
        V[j_offset*4 + k_offset, col] = 1
        W[k_offset*3 + 1, col] = -1  # Z11
        W[k_offset*3 + 2, col] = 1   # Z21
        col += 1
        
        # 7. M7 = x21(y12-y22)
        U[1*4 + j_offset, col] = 1
        V[j_offset*4 + k_offset+1, col] = 1
        V[(j_offset+1)*4 + k_offset+1, col] = -1
        W[k_offset*3 + 1, col] = 1   # Z11
        W[(k_offset+1)*3 + 1, col] = 1 # Z12
        col += 1
        
        # 8. M8 = x32(y21-y11)
        U[2*4 + j_offset+1, col] = 1
        V[(j_offset+1)*4 + k_offset, col] = 1
        V[j_offset*4 + k_offset, col] = -1
        W[k_offset*3 + 2, col] = -1  # Z21
        W[(k_offset+1)*3 + 2, col] = -1 # Z22
        col += 1
        
        # 9. M9 = (x21+x22)y22
        U[1*4 + j_offset, col] = 1
        U[1*4 + j_offset+1, col] = 1
        V[(j_offset+1)*4 + k_offset+1, col] = 1
        W[(k_offset+1)*3 + 1, col] = -1 # Z12
        W[k_offset*3 + 2, col] = -1   # Z21
        W[(k_offset+1)*3 + 2, col] = 1 # Z22
        col += 1
        
        # 10. M10 = (x31-x21)(y11+y12)
        U[2*4 + j_offset, col] = 1
        U[1*4 + j_offset, col] = -1
        V[j_offset*4 + k_offset, col] = 1
        V[j_offset*4 + k_offset+1, col] = 1
        W[k_offset*3 + 1, col] = 1   # Z11
        W[(k_offset+1)*3 + 1, col] = 1 # Z12
        col += 1
        
        # 11. M11 = (x22-x32)(y21+y22)
        U[1*4 + j_offset+1, col] = 1
        U[2*4 + j_offset+1, col] = -1
        V[(j_offset+1)*4 + k_offset, col] = 1
        V[(j_offset+1)*4 + k_offset+1, col] = 1
        W[(k_offset+1)*3 + 1, col] = -1 # Z12
        W[k_offset*3 + 2, col] = 1   # Z21
        W[(k_offset+1)*3 + 2, col] = -1 # Z22
        col += 1
