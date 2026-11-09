    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                edges.append([u, v])
