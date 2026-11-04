    # Add 4x4x4 core terms (remap 4x4 indices to 5x5 indices)
    for (u16, v16, w16) in terms_4x4:
        for i4 in range(4):
            for j4 in range(4):
                idx4 = i4*4 + j4
                if u16[idx4] != 0:
                    U[i4*5 + j4][col] += u16[idx4]
        for i4 in range(4):
            for j4 in range(4):
                idx4 = i4*4 + j4
                if v16[idx4] != 0:
                    V[i4*5 + j4][col] += v16[idx4]
        for i4 in range(4):
            for k4 in range(4):
                idx4 = k4*4 + i4  # W index = k*n + i in 4x4
                if w16[idx4] != 0:
                    W[k4*5 + i4][col] += w16[idx4]
        col += 1
