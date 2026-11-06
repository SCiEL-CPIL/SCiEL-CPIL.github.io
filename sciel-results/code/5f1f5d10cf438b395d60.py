N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            if j < 8:
                # Both in A (0-7)
                colors[idx] = j
            elif i >= 8:
                # Both in B (8-15)
                colors[idx] = j - 8 + 8  # 8..14
            else:
                # i in A, j in B
                colors[idx] = i + 15  # 15..22
            idx += 1
    return colors
# EVOLVE-BLOCK-END
