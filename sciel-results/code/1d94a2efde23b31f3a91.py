        # Pick random edge
        while True:
            u1 = random.randint(0, n-1)
            v1 = random.randint(0, n-1)
            if u1 == v1: continue
            if u1 > v1: u1, v1 = v1, u1
            if adj[u1][v1]:
                break
                
        # Pick random non-edge
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2: continue
            if u2 > v2: u2, v2 = v2, u2
            if not adj[u2][v2]:
                break
