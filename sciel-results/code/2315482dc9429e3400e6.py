rank = 16
U = np.zeros((8, 16), dtype=int)
V = np.zeros((16, 16), dtype=int)
W = np.zeros((8, 16), dtype=int)

for t in range(16):
    j = t // 4  # 0..3
    k = t % 4   # 0..3
    # U[i*4+j, t] = 1 for i in {0,1}
    for i in range(2):
        U[i*4 + j, t] = 1
    # V[j*4+k, t] = 1
    V[j*4 + k, t] = 1
    # W[k*2+i, t] = 1 for i in {0,1}
    for i in range(2):
        W[k*2 + i, t] = 1
