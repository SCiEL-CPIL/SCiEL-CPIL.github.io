"""Matrix multiplication 2x4x7 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = 2, 4, 7
    
    # Rank 50 construction:
    # Split m=4 into m1=2, m2=2.
    # Split p=7 into groups: [2, 2, 2, 1]
    # For each 2x2@2x(2): use Strassen (rank 7)
    # For each 2x2@2x(1): use schoolbook (rank 4)
    # Per 2x2@2x7 block: 3*7 + 1*4 = 25
    # Two blocks: 50
    
    # Strassen for 2x2@2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+g)
    # m7 = (b-d)*(f+h)
    # C[0,0] = m1+m4-m5+m7
    # C[0,1] = m3+m5
    # C[1,0] = m2+m4
    # C[1,1] = m1-m2+m3+m6
    
    # Schoolbook for 2x2@2x1:
    # A=[[a,b],[c,d]], B=[[e],[g]]
    # C[0,0] = a*e + b*g
    # C[1,0] = c*e + d*g
    # 4 mults: a*e, b*g, c*e, d*g
    
    U = []
    V = []
    W = []
    
    # Process two blocks: j in {0,1} and j in {2,3}
    for block in range(2):
        j0 = block * 2  # first j index for this block
        j1 = block * 2 + 1  # second j index
        
        # Split p=7 into [2,2,2,1]
        p_groups = [(0,2), (2,4), (4,6), (6,7)]
        
        for (k_start, k_end) in p_groups:
            group_size = k_end - k_start
            
            if group_size == 2:
                # Strassen: 7 mults
                # A entries: A[0,j0], A[0,j1], A[1,j0], A[1,j1]
                # B entries: B[j0,k_start], B[j0,k_start+1], B[j1,k_start], B[j1,k_start+1]
                # 
                # m1 = (A[0,j0]+A[1,j0])*(B[j0,k_start]+B[j1,k_start])
                # m2 = (A[1,j0]+A[1,j1])*B[j0,k_start]
                # m3 = A[0,j0]*(B[j0,k_start+1]-B[j1,k_start+1])
                # m4 = A[1,j1]*(B[j1,k_start]-B[j0,k_start])
                # m5 = (A[0,j0]+A[0,j1])*B[j1,k_start+1]
                # m6 = (A[1,j0]-A[0,j0])*(B[j0,k_start]+B[j1,k_start])
                # m7 = (A[0,j1]-A[1,j1])*(B[j0,k_start+1]+B[j1,k_start+1])
                
                # Output: C[0,k_start] = m1+m4-m5+m7
                #         C[0,k_start+1] = m3+m5
                #         C[1,k_start] = m2+m4
                #         C[1,k_start+1] = m1-m2+m3+m6
                
                # m1
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j0] = 1; U[col][1*m+j0] = 1
                V[col][j0*p+k_start] = 1; V[col][j1*p+k_start] = 1
                W[col][k_start*n+0] = 1; W[col][k_start*n+1] = 1
                # C[0,k_start] += 1, C[0,k_start+1] += 0, C[1,k_start] += 0, C[1,k_start+1] += 1
                
                # m2
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][1*m+j0] = 1; U[col][1*m+j1] = 1
                V[col][j0*p+k_start] = 1
                W[col][k_start*n+1] = 1
                # C[1,k_start] += 1
                
                # m3
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j0] = 1
                V[col][j0*p+k_start+1] = 1; V[col][j1*p+k_start+1] = -1
                W[col][k_start*n+0] = 1; W[col][k_start+1*n+0] = 1
                # C[0,k_start] += 1, C[0,k_start+1] += 1
                
                # m4
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][1*m+j1] = 1
                V[col][j1*p+k_start] = 1; V[col][j0*p+k_start] = -1
                W[col][k_start*n+0] = 1; W[col][k_start*n+1] = 1
                # C[0,k_start] += 1, C[1,k_start] += 1
                
                # m5
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j0] = 1; U[col][0*m+j1] = 1
                V[col][j1*p+k_start+1] = 1
                W[col][k_start*n+0] = -1; W[col][k_start+1*n+0] = 1
                # C[0,k_start] -= 1, C[0,k_start+1] += 1
                
                # m6
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][1*m+j0] = 1; U[col][0*m+j0] = -1
                V[col][j0*p+k_start] = 1; V[col][j1*p+k_start] = 1
                W[col][k_start+1*n+1] = 1
                # C[1,k_start+1] += 1
                
                # m7
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j1] = 1; U[col][1*m+j1] = -1
                V[col][j0*p+k_start+1] = 1; V[col][j1*p+k_start+1] = 1
                W[col][k_start*n+0] = 1; W[col][k_start+1*n+1] = 1
                # C[0,k_start] += 1, C[1,k_start+1] += 1
                
            else:  # group_size == 1
                # Schoolbook: 4 mults
                # C[0,k_start] = A[0,j0]*B[j0,k_start] + A[0,j1]*B[j1,k_start]
                # C[1,k_start] = A[1,j0]*B[j0,k_start] + A[1,j1]*B[j1,k_start]
                
                # A[0,j0]*B[j0,k_start]
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j0] = 1
                V[col][j0*p+k_start] = 1
                W[col][k_start*n+0] = 1
                
                # A[0,j1]*B[j1,k_start]
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][0*m+j1] = 1
                V[col][j1*p+k_start] = 1
                W[col][k_start*n+0] = 1
                
                # A[1,j0]*B[j0,k_start]
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][1*m+j0] = 1
                V[col][j0*p+k_start] = 1
                W[col][k_start*n+1] = 1
                
                # A[1,j1]*B[j1,k_start]
                col = len(U)
                U.append([0]*(n*m)); V.append([0]*(m*p)); W.append([0]*(p*n))
                U[col][1*m+j1] = 1
                V[col][j1*p+k_start] = 1
                W[col][k_start*n+1] = 1
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
