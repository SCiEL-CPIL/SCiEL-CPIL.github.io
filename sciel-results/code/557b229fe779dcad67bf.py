    def _laderman_3x3x3():
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # Helper to set U[i*3+j][c] = val
        def set_U(i, j, c, val): U[i*3+j][c] = val
        def set_V(j, k, c, val): V[j*3+k][c] = val
        def set_W(k, i, c, val): W[k*3+i][c] = val
        
        # M1
        set_U(0,0,0,1); set_U(1,1,0,1); set_U(2,2,0,1)
        set_V(0,0,0,1); set_V(1,1,0,1); set_V(2,2,0,1)
        set_W(0,0,0,1); set_W(1,1,0,1); set_W(2,2,0,1)
        
        # M2
        set_U(1,0,1,1); set_U(2,0,1,1)
        set_V(0,0,1,1); set_V(1,0,1,1); set_V(2,0,1,1)
        set_W(0,1,1,1); set_W(0,2,1,1)
        
        # M3
        set_U(0,0,2,1); set_U(2,0,2,1)
        set_V(0,0,2,1); set_V(2,0,2,1)
        set_W(0,0,2,1); set_W(0,2,2,1)
        
        # ... and so on for all 23 terms ...
        return U, V, W
    