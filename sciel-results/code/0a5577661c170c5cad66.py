def count_c4s(colors, n):
    # adj[c][i] is bitmask of neighbors
    adj = [[0]*n for _ in range(5)]
    for i in range(n):
        for j in range(i+1, n):
            e = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
            c = colors[e]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    
    total = 0
    for c in range(5):
        for i in range(n):
            for j in range(i+1, n):
                common = adj[c][i] & adj[c][j]
                k = bin(common).count('1')
                if k >= 2:
                    total += k * (k - 1) // 2
    return total
