perm = []
for i1 in range(2):
    for i2 in range(2):
        for j1 in range(2):
            for j2 in range(2):
                R = 8*i1 + 4*i2 + 2*j1 + j2
                K = 8*i1 + 4*j1 + 2*i2 + j2
                perm.append((R, K))
# Create array
p = np.zeros(16, dtype=int)
for R, K in perm:
    p[R] = K
