for i in range(n):
    for k in range(p):
        for j in range(m):
            s = 0
            for t in range(rank):
                s += Um[i*m+j][t] * Vm[j*p+k][t] * Wm[k*n+i][t]
            if abs(s - 1) > 1e-9:
                return False
