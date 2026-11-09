    for r in range(9):
        U[r][:rank_333] = U3[r]
        V[r][:rank_333] = V3[r]  # B[0:3, 0:3]
        W[r][:rank_333] = W3[r]  # C[0:3, 0:3]
