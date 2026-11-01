# M1 = (A00 + A11) * (B0,j1 + B1,j2)
U[0*m+0][col] += 1
U[1*m+1][col] += 1
V[0*p+j1][col] += 1
V[1*p+j2][col] += 1
W[j1*n+0][col] += 1
W[j2*n+1][col] += 1
