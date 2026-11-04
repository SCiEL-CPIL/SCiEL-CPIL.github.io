def _construct_4x4x4_rank28():
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(16)]
    W = [[0]*28 for _ in range(16)]
    
    col = 0
    
    def add(u, v, w):
        nonlocal col
        for idx, val in u:
            U[idx][col] += val
        for idx, val in v:
            V[idx][col] += val
        for idx, val in w:
            W[idx][col] += val
        col += 1
    
    # M1 = (A11+A22)(B11+B22)
    # Contributes to C11 (top-left 2x2 of C)
    for a in range(2):  # row in C11
        for b in range(2):  # col in C11
            # C[a,b] += sum_x (A[a,x]+A[a+2,x+2])*(B[x,b]+B[x+2,b+2])
            # = (A[a,0]+A[a+2,2])*(B[0,b]+B[2,b+2]) + (A[a,1]+A[a+2,3])*(B[1,b]+B[3,b+2])
            # Each product is one rank-1 term in the block sense
            # Term: (A[a,0]+A[a+2,2]) * (B[0,b]+B[2,b+2])
            add([(a*4+0,1),((a+2)*4+2,1)], [(0*4+b,1),(2*4+(b+2),1)], [(a*4+b,1)])
            # Term: (A[a,1]+A[a+2,3]) * (B[1,b]+B[3,b+2])
            add([(a*4+1,1),((a+2)*4+3,1)], [(1*4+b,1),(3*4+(b+2),1)], [(a*4+b,1)])
    
    # M2 = (A21+A22)B11
    # Contributes to C21 (bottom-left 2x2 of C)
    for a in range(2):  # row in C21, actual row = a+2
        for b in range(2):  # col in C21, actual col = b
            # C[a+2,b] += sum_x (A[a+2,x]+A[a+2,x+2])*(B[x,b])
            # = (A[a+2,0]+A[a+2,2])*(B[0,b]) + (A[a+2,1]+A[a+2,3])*(B[1,b])
            add([((a+2)*4+0,1),((a+2)*4+2,1)], [(0*4+b,1)], [(b*4+(a+2),1)])
            add([((a+2)*4+1,1),((a+2)*4+3,1)], [(1*4+b,1)], [(b*4+(a+2),1)])
    
    # M3 = A11(B12-B22)
    # Contributes to C12 (top-right 2x2 of C)
    for a in range(2):  # row in C12
        for b in range(2):  # col in C12, actual col = b+2
            # C[a,b+2] += sum_x A[a,x]*(B[x,b+2]-B[x+2,b+2])
            # = A[a,0]*(B[0,b+2]-B[2,b+2]) + A[a,1]*(B[1,b+2]-B[3,b+2])
            add([(a*4+0,1)], [(0*4+(b+2),1),(2*4+(b+2),-1)], [(b*4+a,1)])
            add([(a*4+1,1)], [(1*4+(b+2),1),(3*4+(b+2),-1)], [(b*4+a,1)])
    
    # M4 = A22(B21-B11)
    # Contributes to C11 and C22
    # C11: C[a,b] += sum_x A[a+2,x+2]*(B[x+2,b]-B[x,b])
    # C22: C[a+2,b+2] += sum_x A[a+2,x+2]*(B[x+2,b+2]-B[x,b+2])
    for a in range(2):
        for b in range(2):
            # C[a,b] += (A[a+2,2]+A[a+2,3])*(B[2,b]-B[0,b]) + ... 
            # Actually: C[a,b] += sum_x A[a+2,x+2]*(B[x+2,b]-B[x,b])
            # = A[a+2,2]*(B[2,b]-B[0,b]) + A[a+2,3]*(B[3,b]-B[1,b])
            add([((a+2)*4+2,1)], [(2*4+b,1),(0*4+b,-1)], [(b*4+a,1)])
            add([((a+2)*4+3,1)], [(3*4+b,1),(1*4+b,-1)], [(b*4+a,1)])
            # C[a+2,b+2] += A[a+2,2]*(B[2,b+2]-B[0,b+2]) + A[a+2,3]*(B[3,b+2]-B[1,b+2])
            add([((a+2)*4+2,1)], [(2*4+(b+2),1),(0*4+(b+2),-1)], [((b+2)*4+(a+2),1)])
            add([((a+2)*4+3,1)], [(3*4+(b+2),1),(1*4+(b+2),-1)], [((b+2)*4+(a+2),1)])
    
    # M5 = (A11+A12)B22
    # Contributes to C11 and C22
    # C11: C[a,b] += sum_x (A[a,x]+A[a,x+2])*(B[x+2,b+2])
    # C22: C[a+2,b+2] += sum_x (A[a+2,x]+A[a+2,x+2])*(B[x+2,b+2])
    # Wait, M5 = (A11+A12)B22, so:
    # C11 += M5, C22 -= M5
    # C[a,b] += sum_x (A[a,x]+A[a,x+2])*(B[x+2,b+2])
    # C[a+2,b+2] -= sum_x (A[a+2,x]+A[a+2,x+2])*(B[x+2,b+2])
    for a in range(2):
        for b in range(2):
            # C[a,b] += (A[a,0]+A[a,2])*(B[2,b+2]) + (A[a,1]+A[a,3])*(B[3,b+2])
            add([(a*4+0,1),(a*4+2,1)], [(2*4+(b+2),1)], [(b*4+a,1)])
            add([(a*4+1,1),(a*4+3,1)], [(3*4+(b+2),1)], [(b*4+a,1)])
            # C[a+2,b+2] -= (A[a+2,0]+A[a+2,2])*(B[2,b+2]) + (A[a+2,1]+A[a+2,3])*(B[3,b+2])
            add([((a+2)*4+0,1),((a+2)*4+2,1)], [(2*4+(b+2),1)], [((b+2)*4+(a+2),-1)])
            add([((a+2)*4+1,1),((a+2)*4+3,1)], [(3*4+(b+2),1)], [((b+2)*4+(a+2),-1)])
    
    # M6 = (A21-A11)(B11+B12)
    # Contributes to C22
    # C[a+2,b+2] += sum_x (A[a+2,x]-A[a,x])*(B[x,b]+B[x,b+2])
    for a in range(2):
        for b in range(2):
            # C[a+2,b+2] += (A[a+2,0]-A[a,0])*(B[0,b]+B[0,b+2]) + (A[a+2,1]-A[a,1])*(B[1,b]+B[1,b+2])
            add([((a+2)*4+0,1),(a*4+0,-1)], [(0*4+b,1),(0*4+(b+2),1)], [((b+2)*4+(a+2),1)])
            add([((a+2)*4+1,1),(a*4+1,-1)], [(1*4+b,1),(1*4+(b+2),1)], [((b+2)*4+(a+2),1)])
    
    # M7 = (A12-A22)(B21+B22)
    # Contributes to C11
    # C[a,b] += sum_x (A[a,x+2]-A[a+2,x+2])*(B[x+2,b]+B[x+2,b+2])
    for a in range(2):
        for b in range(2):
            # C[a,b] += (A[a,2]-A[a+2,2])*(B[2,b]+B[2,b+2]) + (A[a,3]-A[a+2,3])*(B[3,b]+B[3,b+2])
            add([(a*4+2,1),((a+2)*4+2,-1)], [(2*4+b,1),(2*4+(b+2),1)], [(b*4+a,1)])
            add([(a*4+3,1),((a+2)*4+3,-1)], [(3*4+b,1),(3*4+(b+2),1)], [(b*4+a,1)])
    
    return U, V, W
