# A_R (2x2) * B_B (2x4): Split B_B into two 2x2 blocks by columns
for block in range(2):
    k_start = block * 2
    ...
    U[li*4 + 2 + lj, col] = sU[li*2 + lj, c]
