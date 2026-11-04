s = 0.0
for c in range(U.shape[1]):
    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
tgt = 1.0 if i == k else 0.0
