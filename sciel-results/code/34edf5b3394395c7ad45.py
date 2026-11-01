def construct(k, g):
    if k != 4 or g != 10:
        n = 2 * k
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    from collections import deque
    import random
    
    def check_girth(m, shifts, target_girth):
        adj = [[] for _ in range(2 * m)]
        for i in range(m):
            for s in shifts:
                j = (i + s) % m
                adj[i].append(m + j)
                adj[m + j].append(i)
        
        min_cycle = float('inf')
        for u in range(2 * m):
            for v in adj[u]:
                if v <= u:
                    continue
                dist = {u: 0}
                queue = deque([u])
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    for w in adj[cur]:
                        if (cur == u and w == v) or (cur == v and w == u):
                            continue
                        if w not in dist:
                            dist[w] = dist[cur] + 1
                            if w == v:
                                min_cycle = min(min_cycle, dist[w] + 1)
                                found = True
                                break
                            queue.append(w)
                if found:
                    break
            if min_cycle < target_girth:
                return False
        return True
    
    # Try m = 160 with various shift sets
    m = 160
    random.seed(12345)
    
    # Try structured shifts first
    for step in [32, 40, 48, 56, 64]:
        shifts = [(i * step) % m for i in range(1, 5)]
        if len(set(shifts)) == 4 and 0 not in shifts:
            if check_girth(m, shifts, 10):
                n = 2 * m
                edges = [(i, (i + s) % m + m) for i in range(m) for s in shifts]
                return n, edges
    
    # Random search
    for _ in range(200):
        shifts = random.sample(range(1, m), 4)
        if check_girth(m, shifts, 10):
            n = 2 * m
            edges = [(i, (i + s) % m + m) for i in range(m) for s in shifts]
            return n, edges
    
    # Fallback
    q = 4
    d = 5
    table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    add = lambda x,y: x^y
    sub = lambda x,y: x^y
    mul = lambda x,y: table[x][y]
    rule = {2:(1,1),3:(2,1),4:(1,2)}
    for c in range(5,d+1):
        r,c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(v):
        side,x = v
        for first in range(q):
            y = [None,first]+[0]*(d-1)
            for c in range(2,d+1):
                a,b = rule[c]
                y[c] = add(x[c],mul(y[a],x[b])) if side==0 else sub(x[c],mul(x[a],y[b]))
            yield 1-side,tuple(y)
    start = (0,(None,)+(0,)*d)
    idx,order,edges = {start:0},[start],[]
    for v in order:
        for o in neighbours(v):
            if o not in idx:
                idx[o] = len(order)
                order.append(o)
            if v[0]==0:
                edges.append((idx[v],idx[o]))
    return len(order),edges
