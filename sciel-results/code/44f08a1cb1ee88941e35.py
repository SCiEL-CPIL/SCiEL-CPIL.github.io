def construct():
    import random
    
    def make_adj(p, residues):
        adj = [[False] * p for _ in range(p)]
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in residues:
                    adj[i][j] = adj[j][i] = True
        return adj
    
    def count_violations_adj(adj, n):
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            if (adj[i][j] and adj[i][k] and adj[i][l] and adj[i][m] and
                                adj[j][k] and adj[j][l] and adj[j][m] and
                                adj[k][l] and adj[k][m] and adj[l][m]):
                                cnt += 1
                            elif (not adj[i][j] and not adj[i][k] and not adj[i][l] and not adj[i][m] and
                                  not adj[j][k] and not adj[j][l] and not adj[j][m] and
                                  not adj[k][l] and not adj[k][m] and not adj[l][m]):
                                cnt += 1
        return cnt
    
    def local_search(adj, n, max_iters=20000):
        random.seed(12345)
        current_v = count_violations_adj(adj, n)
        for it in range(max_iters):
            if current_v == 0:
                return True
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            # Flip edge (i,j)
            adj[i][j] = adj[j][i] = not adj[i][j]
            new_v = count_violations_adj(adj, n)
            if new_v < current_v:
                current_v = new_v
            else:
                adj[i][j] = adj[j][i] = not adj[i][j]
        return current_v == 0
    
    for p in [41, 37]:
        if p % 4 != 1:
            continue
        residues = {(x * x) % p for x in range(1, p)}
        adj = make_adj(p, residues)
        if local_search(adj, p, max_iters=5000):
            edges = [(i, j) for i in range(p) for j in range(i+1, p) if adj[i][j]]
            return p, edges
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i+1, q) if (j - i) % q in residues]
