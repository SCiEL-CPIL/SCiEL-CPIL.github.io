for i in range(2):
    for dj in range(2):
        coeff = ac[i*2+dj]
        if coeff != 0:
            j = a*2 + dj
            U[i*m + j][col] += coeff

for dj in range(2):
    for dk in range(2):
        coeff = bc[dj*2+dk]
        if coeff != 0:
            j = a*2 + dj
            k = b*2 + dk
            V[j*p + k][col] += coeff

for i in range(2):
    for dk in range(2):
        coeff = cc[i*2+dk]
        if coeff != 0:
            k = b*2 + dk
            W[k*n + i][col] += coeff
