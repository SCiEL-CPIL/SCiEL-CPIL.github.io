<<<<<<< SEARCH
    U[0][0] = 1   # a in M1
    U[3][0] = 1   # d in M1
    U[2][1] = 1   # c in M2
    U[3][1] = 1   # d in M2
    U[0][2] = 1   # a in M3
    U[3][3] = 1   # d in M4
    U[0][4] = 1   # a in M5
    U[1][4] = 1   # b in M5
    U[2][5] = 1   # c in M6
    U[0][5] = -1  # a in M6 (negative)
    U[1][6] = 1   # b in M7
    U[3][6] = -1  # d in M7 (negative)
    
    # V factors (row = j*p+k, col = r):
    # r=0: e(1), h(1) -> V[0][0]=1, V[3][0]=1
    # r=1: e(1) -> V[0][1]=1
    # r=2: f(1), h(-1) -> V[1][2]=1, V[3][2]=-1
    # r=3: g(1), e(-1) -> V[2][3]=1, V[0][3]=-1
    # r=4: h(1) -> V[3][4]=1
    # r=5: e(1), f(1) -> V[0][5]=1, V[1][5]=1
    # r=6: g(1), h(1) -> V[2][6]=1, V[3][6]=1
    
    V[0][0] = 1   # e in M1
    V[3][0] = 1   # h in M1
    V[0][1] = 1   # e in M2
    V[1][2] = 1   # f in M3
    V[3][2] = -1  # h in M3 (negative)
    V[2][3] = 1   # g in M4
    V[0][3] = -1  # e in M4 (negative)
    V[3][4] = 1   # h in M5
    V[0][5] = 1   # e in M6
    V[1][5] = 1   # f in M6
    V[2][6] = 1   # g in M7
    V[3][6] = 1   # h in M7
    
    # W factors (row = k*n+i, col = r):
    # C[0,0] = M1 + M4 - M5 + M7 -> W[0]
    # C[0,1] = M3 + M5 -> W[2]
    # C[1,0] = M2 + M4 -> W[1]
    # C[1,1] = M1 - M2 + M3 + M6 -> W[3]
    
    # W[0] (C[0,0]): M1(+1), M4(+1), M5(-1), M7(+1)
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7
    
    # W[1] (C[1,0]): M2(+1), M4(+1)
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # W[2] (C[0,1]): M3(+1), M5(+1)
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # W[3] (C[1,1]): M1(+1), M2(-1), M3(+1), M6(+1)
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
=======
    U[0, 0] = 1   # a in M1
    U[3, 0] = 1   # d in M1
    U[2, 1] = 1   # c in M2
    U[3, 1] = 1   # d in M2
    U[0, 2] = 1   # a in M3
    U[3, 3] = 1   # d in M4
    U[0, 4] = 1   # a in M5
    U[1, 4] = 1   # b in M5
    U[2, 5] = 1   # c in M6
    U[0, 5] = -1  # a in M6 (negative)
    U[1, 6] = 1   # b in M7
    U[3, 6] = -1  # d in M7 (negative)
    
    V[0, 0] = 1   # e in M1
    V[3, 0] = 1   # h in M1
    V[0, 1] = 1   # e in M2
    V[1, 2] = 1   # f in M3
    V[3, 2] = -1  # h in M3 (negative)
    V[2, 3] = 1   # g in M4
    V[0, 3] = -1  # e in M4 (negative)
    V[3, 4] = 1   # h in M5
    V[0, 5] = 1   # e in M6
    V[1, 5] = 1   # f in M6
    V[2, 6] = 1   # g in M7
    V[3, 6] = 1   # h in M7
    
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # M5
    W[0, 6] = 1   # M7
    
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4
    
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5
    
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
