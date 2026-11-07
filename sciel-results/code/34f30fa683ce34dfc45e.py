"""Triangle-free diameter-four augmentation: greedy + local search with valid fallback."""
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _triangle_free(edge_set, n):
    adj = _adj(edge_set, n)
    for u in range(n):
        nu = adj[u]
        for v in nu:
            if v <= u:
                continue
            for w in nu:
                if w > v and w in adj[v]:
                    return False
    return True

def _diam(edge_set, n):
    adj = _adj(edge_set, n)
    present = set()
    for u, v in edge_set:
        present.add(u); present.add(v)
    for i in range(n):
        if i not in present:
            return -1
    def bfs(s):
        d = [-1]*n; d[s]=0; q=[s]; h=0
        while h < len(q):
            u=q[h]; h+=1
            for v in adj[u]:
                if d[v]<0:
                    d[v]=d[u]+1; q.append(v)
        return d
    best=0
    for s in range(n):
        d=bfs(s)
        if any(x<0 for x in d):
            return -1
        best=max(best, max(d))
    return best

def _valid(edge_set, n):
    if not _triangle_free(edge_set, n):
        return False
    d=_diam(edge_set, n)
    return d>=0 and d<=4

def construct():
    t0=time.time()
    DEADLINE=78.0
    n=P['n']
    base_pairs=set()
    for e in P['edges']:
        base_pairs.add(frozenset((e[0],e[1])))
    base_pairs=frozenset(base_pairs)

    candidates=[(u,v) for u in range(n) for v in range(u+1,n) if frozenset((u,v)) not in base_pairs]

    def common_nei_count(edge_set, u, v):
        adj=_adj(edge_set,n)
        return len(adj[u] & adj[v])

    cur=set(base_pairs)
    added=[]

    # Greedy add
    for _ in range(80):
        if time.time()>DEADLINE:
            break
        d=_diam(set((a,b) for a,b in cur), n)
        if d<=4:
            break
        best=None; best_d=None
        adj=_adj(set((a,b) for a,b in cur), n)
        for (u,v) in candidates:
            if frozenset((u,v)) in cur:
                continue
            if adj[u] & adj[v]:
                continue
            trial=set(cur); trial.add(frozenset((u,v)))
            td=_diam(set((a,b) for a,b in trial), n)
            if td<0 or td>4:
                continue
            if best_d is None or td<best_d:
                best_d=td; best=(u,v)
        if best is None:
            break
        cur.add(frozenset(best)); added.append(list(best))

    # Local search removal
    for _ in range(40):
        if time.time()>DEADLINE:
            break
        done=False
        for i in range(len(added)):
            if time.time()>DEADLINE:
                break
            e=frozenset(added[i])
            trial=set(cur); trial.discard(e)
            if _valid(set((a,b) for a,b in trial), n):
                cur=trial; added.pop(i); done=True; break
        if not done:
            break

    final=set((a,b) for a,b in cur)
    if not _valid(final, n):
        added=_fallback(n, base_pairs)
    return [list(e) for e in added]

def _fallback(n, base_pairs):
    # Guaranteed valid: build diameter-4 triangle-free supergraph of the path.
    # Use two hubs 11 and 12 (edge 11-12 in base). Connect 11 to left side, 12 to right side via length-2 paths to avoid triangles.
    # Triangle-free: a leaf attached to hub h must not be adjacent to any other neighbor of h.
    # We'll attach each vertex i<11 to hub 11 only if i is not adjacent to 10 or 12 or 0.. via base? i adjacent to i-1,i+1 in base.
    # 11's base neighbors:10,12. So a leaf i attached to 11 forms triangle if i adjacent to 10 or 12. i adjacent to 10 iff i in {9,11}; i adjacent to 12 iff i in {11,13}. So i=9 or 13 would triangle. Avoid those.
    # For diameter, attach vertices to nearest hub so max dist 2 from hub, hubs dist 1 -> max 2+1+2=5? too big. Need hubs close and leaves dist1 from hub -> leaf to other hub = 1+1=2, leaf to leaf same hub=2, opposite=4. Good: diameter 4.
    # So attach each vertex to exactly one hub at distance 1 (direct edge), hubs 11-12.
    # But leaf i attached to 11: triangle if i~10 or i~12 (base). i~10 base iff i=9 or 11. i~12 base iff i=13 or 11. So avoid i in {9,13}.
    # Also leaf i attached to 11 and another leaf j attached to 11: i-j edge base? if |i-j|=1 and both attached to 11, triangle i-11-j. So no two adjacent vertices both attached to same hub.
    # This is a 2-coloring constraint on the path excluding hubs.
    # Path 0..10 left of 11, 12..23 right. 11 and 12 hubs.
    # Left vertices 0..10: attach to 11, but 9 adjacent 10 and 10 adjacent 11(base) so 9~10 base, 10~11 base -> 9 attached 11 and 10 attached 11 => 9-11-10 triangle? 9-10 base yes -> triangle. So 9 and 10 can't both be 11. 10 is adjacent to 11 already (base) so 10 is already "at" 11 distance 1. Attaching 9 to 11: 9-11-10 triangle. So 9 must attach to 12? 9-12 edge: 9 adjacent 8,10; 12 adjacent 11,13. 9~10,12~11: 10 and 11 adjacent base -> 9-12-11? no 9-11 not edge. 9-12 and 12-11 base and 9-11? not edge. ok. But 9 to 12 distance: 9-12 direct =1, then 12-11=1. fine.
    # This is getting intricate. Let me just construct explicitly and verify, returning it only if valid.
    edges=[]
    hubs=[11,12]
    # assign each non-hub vertex to a hub
    assign={}
    for i in range(n):
        if i in hubs: continue
        # choose hub that minimizes triangle risk: prefer hub not adjacent to i's base neighbors
        # base neighbors of i: i-1,i+1 (in range)
        nb=set()
        if i-1>=0: nb.add(i-1)
        if i+1<n: nb.add(i+1)
        # 11 base neighbors:10,12 ; 12 base neighbors:11,13
        # i attached to 11 triangle if any nb in {10,12}
        # i attached to 12 triangle if any nb in {11,13}
        ok11=not (nb & {10,12})
        ok12=not (nb & {11,13})
        if ok11 and not ok12: assign[i]=11
        elif ok12 and not ok11: assign[i]=12
        elif ok11: assign[i]=11
        else: assign[i]=12
    # Now also need: two adjacent vertices assigned same hub => triangle. Fix by flipping one.
    for i in range(n-1):
        if i in hubs or i+1 in hubs: continue
        if assign[i]==assign[i+1]:
            # flip i+1 to other hub if safe
            other=12 if assign[i+1]==11 else 11
            nb=set()
            if i+2<n: nb.add(i+2)
            if i>=0: nb.add(i)
            if other==11 and not (nb & {10,12}):
                assign[i+1]=other
            elif other==12 and not (nb & {11,13}):
                assign[i+1]=other
            else:
                # flip i instead
                nb2=set()
                if i-1>=0: nb2.add(i-1)
                if i+1<n: nb2.add(i+1)
                if assign[i]==11:
                    if not (nb2 & {11,13}): assign[i]=12
                else:
                    if not (nb2 & {10,12}): assign[i]=11
    for i in range(n):
        if i in hubs: continue
        h=assign[i]
        edges.append([i,h])
    # Verify
    es=set(frozenset((a,b)) for a,b in edges)
    full=set(base_pairs)|es
    if _valid(full, n):
        return [list(e) for e in edges]
    # If still invalid, brute small search
    return _brute_fallback(n, base_pairs)

def _brute_fallback(n, base_pairs):
    # Last resort: try adding edges from a candidate pool greedily until valid, then return.
    t0=time.time()
    cur=set(base_pairs)
    pool=[(u,v) for u in range(n) for v in range(u+1,n) if frozenset((u,v)) not in base_pairs]
    added=[]
    for (u,v) in pool:
        if time.time()-t0>50: break
        if frozenset((u,v)) in cur: continue
        adj=_adj(set((a,b) for a,b in cur),n)
        if adj[u]&adj[v]: continue
        trial=set(cur); trial.add(frozenset((u,v)))
        full=set((a,b) for a,b in trial)
        if _valid(full,n):
            cur=trial; added.append([u,v])
            if _diam(set((a,b) for a,b in cur),n)<=4:
                break
    full=set((a,b) for a,b in cur)
    if _valid(full,n):
        return added
    return []
