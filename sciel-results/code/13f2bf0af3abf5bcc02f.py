For j in {0,1,2}:
  For r in {0,1,2} (rank index for this j-block):
    U[i*m+j, col] = 1 if i == r, else 0
    V[j*p+k, col] = 1 for all k (since B[j,k] is multiplied)
    Wait, that's not right either.
