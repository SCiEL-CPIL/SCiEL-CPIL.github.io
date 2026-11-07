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
    else:
        # Filter out any base edges just in case (shouldn't happen, but safety)
        added = [e for e in added if frozenset(e) not in base_pairs]
        
    # Final safety check: ensure all returned edges are valid types and not base
    result = []
    for e in added:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = e
            if isinstance(u, int) and isinstance(v, int) and not isinstance(u, bool) and not isinstance(v, bool):
                if frozenset((u,v)) not in base_pairs:
                    result.append([u, v])
    return result

def _fallback(n, base_pairs):
    # Simple constructive fallback:
    # Hubs 11, 12.
    # We need to add edges to reduce diameter.
    # Base path has diameter 23.
    # Add edges to make it a "star-like" structure around hubs.
    # To be triangle-free, we cannot add edge (u,v) if N(u) and N(v) intersect.
    # In the base path, N(i) = {i-1, i+1}.
    # If we add (i, 11), we cannot have 11 adjacent to i-1 or i+1 in the *current* graph.
    # 11 is adjacent to 10, 12 in base.
    # So if i=9, N(9)={8,10}. 10 is in N(11). Triangle 9-10-11. So cannot add (9,11).
    # If i=10, N(10)={9,11}. 11 is in N(11)? No, self. But 11 is the hub. Edge (10,11) is base.
    # We should only add edges NOT in base.
    
    edges = []
    hubs = [11, 12]
    
    # Try to connect all vertices to one of the hubs if possible.
    # If not possible due to triangles, connect to a "bridge" vertex?
    # Actually, a known valid solution for path P24 with diameter 4:
    # Add edges (0,11), (1,12), (2,11), (3,12), ..., (10,11)?
    # Let's check triangles for (i, 11).
    # N(11) initially {10, 12}.
    # If we add (i, 11), new N(11) includes i.
    # Triangle i-11-j exists if j in N(i) and j in N(11).
    # N(i) in base is {i-1, i+1}.
    # So if i-1 or i+1 is in N(11), triangle.
    # Initially N(11)={10,12}.
    # So i cannot be 9 (i+1=10) or 13 (i-1=12) or 10 (base edge) or 12 (base edge).
    # If we add (9, 12) instead?
    # N(12)={11,13}.
    # N(9)={8,10}. No intersection. So (9,12) is safe.
    # Distances:
    # 0-1-...-9-12-11-10-...
    # Let's just use a greedy approach in the fallback that is fast and correct.
    
    cur = set(base_pairs)
    added = []
    # Candidate edges sorted by potential to reduce diameter (longer jumps first?)
    # Or just iterate all non-base edges.
    candidates = [(u, v) for u in range(n) for v in range(u+1, n) if frozenset((u,v)) not in base_pairs]
    
    # Greedy: add edge if it doesn't create triangle and reduces diameter (or just adds).
    # To ensure termination and validity, we just add edges that keep it triangle-free.
    # We need diameter <= 4.
    
    # Heuristic: connect vertices far from center to center.
    # Center is 11.5.
    # Try adding (i, 11) for i in 0..10, (i, 12) for i in 13..23.
    # Check triangle constraints dynamically.
    
    adj = _adj(cur, n)
    
    # Order candidates by distance from center?
    # Let's just try specific promising edges.
    # Left side 0-10, Right side 13-23.
    # Hubs 11, 12.
    
    # Strategy:
    # 1. Add (i, 11) for i in 0,2,4,6,8? (Even indices to avoid adjacent conflicts?)
    # 2. Add (i, 12) for i in 13,15,17,19,21,23?
    
    # Let's just run a simple greedy on the full candidate list, prioritizing edges that connect far nodes.
    # Sort candidates by (max(u,v) - min(u,v)) descending? No, by distance in path.
    # Path distance is abs(u-v).
    candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    for u, v in candidates:
        if frozenset((u,v)) in cur: continue
        # Check triangle-free
        if adj[u] & adj[v]: continue
        # Add it
        cur.add(frozenset((u,v)))
        added.append([u,v])
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if diameter is <= 4
        # Fast check: BFS from all nodes?
        # n=24, so 24 BFSs is cheap.
        d = _diam(set(cur), n)
        if d <= 4:
            break
            
    # Verify final validity
    if _valid(set(cur), n):
        # Ensure no base edges in added
        return [e for e in added if frozenset(e) not in base_pairs]
        
    return _brute_fallback(n, base_pairs)

def _brute_fallback(n, base_pairs):
    # Last resort: try adding edges from a candidate pool greedily until valid, then return.
    t0=time.time()
    cur=set(base_pairs)
    # Prioritize edges that likely reduce diameter: long range
    pool=[(u,v) for u in range(n) for v in range(u+1,n) if frozenset((u,v)) not in base_pairs]
    pool.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    added=[]
    for (u,v) in pool:
        if time.time()-t0>50: break
        if frozenset((u,v)) in cur: continue
        adj=_adj(set((a,b) for a,b in cur),n)
        if adj[u]&adj[v]: continue
        trial=set(cur); trial.add(frozenset((u,v)))
        # Quick triangle check already done by adj intersection.
        # Check diameter
        if _diam(set((a,b) for a,b in trial),n) <= 4:
            cur=trial; added.append([u,v])
            break # Found a valid solution
    full=set((a,b) for a,b in cur)
    if _valid(full,n):
        return [e for e in added if frozenset(e) not in base_pairs]
    return []
