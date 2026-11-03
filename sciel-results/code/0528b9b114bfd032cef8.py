    def get_delta(a, b, p, i, j):
        u = p[i]
        v = p[j]
        delta = 0
        # Self
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        # Cross
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        
        # Others
        for k in range(len(p)):
            if k == i or k == j:
                continue
            pk = p[k]
            # i interacts with k
            delta += a[i, k] * (b[v, pk] - b[u, pk])
            # k interacts with i
            delta += a[k, i] * (b[pk, v] - b[pk, u])
            # j interacts with k
            delta += a[j, k] * (b[u, pk] - b[v, pk])
            # k interacts with j
            delta += a[k, j] * (b[pk, u] - b[pk, v])
            
        return delta
    