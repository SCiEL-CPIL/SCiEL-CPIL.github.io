for t in range(7):
    # U access: strassen_U[t][i*m+j] where i,j in 0..1, so indices 0..3 ✓
    # V access: strassen_V[t][0..3] ✓
    # W access: strassen_W[t][0..3] - but t goes 0..6, and strassen_W only has 4 rows!
