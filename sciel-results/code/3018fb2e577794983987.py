U4 = np.zeros((16, 49))
for i1 in range(2):
    for i2 in range(2):
        for j1 in range(2):
            for j2 in range(2):
                row4 = i1*8 + i2*4 + j1*2 + j2
                row2_outer = i1*2 + j1
                row2_inner = i2*2 + j2
                for r1 in range(7):
                    for r2 in range(7):
                        r = r1*7 + r2
                        U4[row4, r] = U2[row2_outer, r1] * U2[row2_inner, r2]
