# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    
    # Strassen decomposition for 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Terms: (U_dict, V_dict, W_dict)
    # Indices: U[i,j], V[j,k], W[k,i]
    # W index in output matrix: k*n + i
    
    # M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    # M2 = (c+d)e    -> C[1,0]+=1, C[1,1]-=1
    # M3 = a(f-h)    -> C[0,1]+=1, C[1,1]+=1
    # M4 = d(g-e)    -> C[0,0]+=1, C[1,0]+=1
    # M5 = (a+b)h    -> C[0,0]-=1, C[0,1]+=1
    # M6 = (c-a)(e+f)-> C[1,1]+=1
    # M7 = (b-d)(g+h)-> C[0,0]+=1
    
    terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    rank = num_blocks * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for b in range(num_blocks):
        for t in range(7):
            col = b*7 + t
            u_t, v_t, w_t = terms[t]
            
            for (i,j), val in u_t.items():
                U[i*m + j][col] = val
            
            for (j,k), val in v_t.items():
                V[j*p + (2*b + k)][col] = val
            
            for (k,i), val in w_t.items():
                W[(2*b + k)*n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
