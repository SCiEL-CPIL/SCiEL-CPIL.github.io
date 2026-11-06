def construct():
    N = 16
    colors = [0] * (N*(N-1)//2)
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i*(N-1) - i*(i-1)//2 + (j - i - 1)

    for i in range(N-1):
        for j in range(i+1, N):
            if i == 14 and j == 15:
                colors[edge_idx(i, j)] = 15
            else:
                colors[edge_idx(i, j)] = i
                
    return colors
