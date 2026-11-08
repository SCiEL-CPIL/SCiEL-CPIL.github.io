# 4x4x1 part: single term
U[i*5+j][col] = 1 for all i=0..3, j=0..3
V[j*5+4][col] = 1 for all j=0..3
W[4*4+i][col] = 1 for all i=0..3
col += 1

# 4x1x4 part: single term
U[i*5+4][col] = 1 for all i=0..3
V[4*5+k][col] = 1 for all k=0..3
W[k*4+i][col] = 1 for all i=0..3
col += 1

# 4x1x1 part: single term
U[i*5+4][col] = 1 for all i=0..3
V[4*5+4][col] = 1
W[4*4+i][col] = 1 for all i=0..3
col += 1
