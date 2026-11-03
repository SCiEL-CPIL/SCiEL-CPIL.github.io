def _build_4x4x4_strassen():
    # ... existing code ...
    pass

def construct():
    # ...
    # Build 4x4x4 Strassen (Rank 28)
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Verify 4x4x4 part
    if _verify_decomposition_fast(U44, V44, W44, 4, 4, 4):
        # Extend to 4x4x5
        # Rank 28 + 1 = 29
        rank_total = 29
        U = [[0]*rank_total for _ in range(16)]
        V = [[0]*rank_total for _ in range(20)]
        W = [[0]*rank_total for _ in range(20)]
        
        # Copy Strassen part
        for c in range(28):
            for i in range(4):
                for j in range(4):
                    U[i*4+j][c] = U44[i*4+j][c]
            for j in range(4):
                for k in range(4):
                    V[j*5+k][c] = V44[j*4+k][c]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][c] = W44[k*4+i][c]
        
        # Add boundary term c=28
        # U[i,j,28] = 1
        for i in range(4):
            for j in range(4):
                U[i*4+j][28] = 1
        
        # V[j,4,28] = 1
        for j in range(4):
            V[j*5+4][28] = 1
        
        # W[4,i,28] = 1
        for i in range(4):
            W[4*4+i][28] = 1
            
        # Verify full 4x4x5
        if _verify_decomposition_fast(U, V, W, 4, 4, 5):
             return {"U": U, "V": V, "W": W}
             
    # Fallback to schoolbook if Strassen fails
    U_school, V_school, W_school = _schoolbook(4, 4, 5)
    return {"U": U_school, "V": V_school, "W": W_school}
