def compute_alpha(edges, n=20):
    # edges: list of frozensets or tuples of 3 vertices
    # Precompute
    edges_mask = [0]*len(edges)
    for i,e in enumerate(edges):
        m = 0
        for v in e: m |= 1<<v
        edges_mask[i] = m
    # pair index
    pair_idx = {}
    pair_endpoints = []
    idx = 0
    for i in range(n):
        for j in range(i+1,n):
            pair_idx[(i,j)] = idx
            pair_endpoints.append((i,j))
            idx += 1
    # edges_with[v] = list of (u,w,pi)
    edges_with = [[] for _ in range(n)]
    for e in edges:
        a,b,c = sorted(e)
        edges_with[a].append((b,c,pair_idx[(b,c)]))
        edges_with[b].append((a,c,pair_idx[(a,c)]))
        edges_with[c].append((a,b,pair_idx[(a,b)]))
    
    best = 0
    nodes = 0
    import sys
    sys.setrecursionlimit(10000)
    
    def popcount(x): return bin(x).count('1')
    
    def upper_bound(A):
        # maximal matching size
        rem = A
        cnt = 0
        for em in edges_mask:
            if em & rem == em:
                cnt += 1
                rem &= ~em
        return popcount(A) - cnt
    
    def has_edge(A):
        for em in edges_mask:
            if em & A == em:
                return True
        return False
    
    def search(A, F, chosen):
        nonlocal best, nodes
        nodes += 1
        a = popcount(A)
        # special: no edges in A
        if not has_edge(A):
            val = chosen + a
            if val > best: best = val
            return
        ub = chosen + upper_bound(A)
        if ub <= best:
            return
        if a <= 1:
            val = chosen + a
            if val > best: best = val
            return
        # pick vertex in most edges within A
        best_v = -1
        best_cnt = -1
        av = A
        while av:
            lsb = av & (-av)
            v = lsb.bit_length()-1
            cnt = 0
            for (u,w,pi) in edges_with[v]:
                if (A>>u)&1 and (A>>w)&1:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best_v = v
            av &= av-1
        v = best_v
        # Branch 1: exclude v
        A1 = A & ~(1<<v)
        # restrict F to pairs within A1
        F1 = set()
        for pi in F:
            i,j = pair_endpoints[pi]
            if (A1>>i)&1 and (A1>>j)&1:
                F1.add(pi)
        search(A1, F1, chosen)
        # Branch 2: include v
        A2 = A & ~(1<<v)
        # remove forbidden partners of v
        for pi in F:
            i,j = pair_endpoints[pi]
            if i==v:
                A2 &= ~(1<<j)
            elif j==v:
                A2 &= ~(1<<i)
        # new F: restrict to A2, plus new pairs from edges with v
        F2 = set()
        for pi in F:
            i,j = pair_endpoints[pi]
            if (A2>>i)&1 and (A2>>j)&1:
                F2.add(pi)
        for (u,w,pi) in edges_with[v]:
            if (A2>>u)&1 and (A2>>w)&1:
                F2.add(pi)
        search(A2, F2, chosen+1)
    
    search((1<<n)-1, set(), 0)
    return best
