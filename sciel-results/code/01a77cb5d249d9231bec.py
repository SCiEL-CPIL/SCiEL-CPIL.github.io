    # Correct V mapping
    V = np.zeros((28, total_rank))
    for j in range(4):
        for l in range(4):
            V[j * 7 + l, 0:rank_444] = V444[j * 4 + l, 0:rank_444]
