for i2 in range(2):
    for k2 in range(2):
        i = bi * 2 + i2
        k = bj * 2 + k2
        W4[k*4+i][c] += Wcoeff[s][bi][bj] * W2[k2*2+i2][t]
