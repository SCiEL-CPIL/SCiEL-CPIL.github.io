N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N * (N - 1) // 2)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if i == 0:
                c = j
            else:
                c = 0
            colors[idx] = c
            idx += 1
    return colors
# EVOLVE-BLOCK-END
