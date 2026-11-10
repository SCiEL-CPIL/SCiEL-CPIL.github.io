"""Dense odd-order graph with all-distinct degrees -> constraint vacuously satisfied."""
import time

N = 15

def _hh(deg_in):
    """Havel-Hakimi: return edge list or None if not graphical."""
    # work with (deg, vertex) pairs
    arr = list(range(N))
    d = list(deg_in)
    edges = []
    for step in range(N):
        # find vertex with max degree
        m = max(d)
        if m < 0:
            break
        idx = d.index(m)
        d[idx] = -1
        if m == 0:
            continue
        # connect to next m vertices with highest remaining degree
        rem = [(d[j], j) for j in range(N) if d[j] > 0]
        rem.sort(reverse=True)
        if len(rem) < m:
            return None
        for k in range(m):
            j = rem[k][1]
            edges.append((idx, j))
            d[j] -= 1
        if any(x < 0 for x in d):
            return None
    return edges

def construct():
    deadline = time.time() + 80.0

    # Base: degrees 0..14 (sum 105 odd -> not graphical!).
    # Need sum even. Use degrees {0,1,2,...,13,14} sum=105 odd.
    # Adjust: use {1,2,3,4,5,6,7,8,9,10,11,12,13,14,14}? not distinct.
    # Distinct 15 values in [0,14] with even sum: 0..14 sum=105 odd.
    # Replace one to make even: e.g. drop 0, add 15? max is 14.
    # Use {0,1,2,...,13, 14} = 105. Swap 0->1 and 1->0? same set.
    # Try {1,2,...,14, 14} no. We need 15 distinct in [0,14] -> exactly 0..14.
    # 105 is odd -> NOT graphical. So all-distinct 0..14 impossible!
    #
    # Use a different distinct set with even sum within [0,14]:
    # 15 distinct values from 0..14 = only {0..14}. Sum 105 odd.
    # So we CANNOT have all 15 degrees distinct AND graphical with these bounds?
    # Actually graphical sequences need even sum. 0..14 fails.
    #
    # Alternative: don't require all distinct; require that no two vertices
    # that can be 3-path endpoints share a degree. That's the real constraint.
    # But simplest safe target: make degrees as distinct as possible.
    #
    # Reconsider: maybe 0..14 with even sum by using a multiset is not allowed
    # (distinct). So all-distinct is impossible for N=15 (sum must be even).
    #
    # New plan: use distinct degrees EXCEPT allow the constraint directly.
    # Simpler robust approach: maximize edges under the actual 3-path constraint
    # via greedy + local search, verified exactly.

    # Greedy edge addition with exact constraint check.
    adj = [set() for _ in range(N)]
    edges = []

    def deg(i):
        return len(adj[i])

    def valid_after(u, v):
        # adding edge (u,v): check every 3-path s-u-v-t and s-v-u-t has deg(s)!=deg(t)
        # but degrees change for u and v only.
        # A 3-path endpoint equality: for path s-u-v-t, endpoints s and t.
        # deg(s), deg(t) in the NEW graph. u,v degrees increase by 1.
        # Check all pairs (s,t) with s in adj[u], t in adj[v], s!=t, s!=v, t!=u,
        # and s!=t, and also s not adjacent to v? No: path s-u-v-t is simple 3-edge
        # path, need s-u, u-v, v-t edges and s!=t, s!=v, t!=u. s and t are endpoints.
        # After adding u-v, deg_new(u)=deg(u)+1, deg_new(v)=deg(v)+1, others same.
        # Constraint: for ALL such paths, deg_new(s) != deg_new(t).
        # But this must hold for the WHOLE graph after addition, i.e. every 3-path,
        # not just those using new edge. Adding an edge can only CREATE new 3-paths
        # that use (u,v) as the middle edge. Existing 3-paths unaffected (their
        # endpoint degrees unchanged since only u,v degrees change, and if a
        # pre-existing 3-path had endpoint u or v, its degree changed -> could break!).
        #
        # So we must check: after adding (u,v), for EVERY 3-path s-a-b-t in the new
        # graph, deg_new(s) != deg_new(t).
        # This is expensive but N=15 small.

        # Build new adjacency
        newadj = [set(x) for x in adj]
        newadj[u].add(v); newadj[v].add(u)
        nd = [len(x) for x in newadj]
        for a in range(N):
            for b in newadj[a]:
                if a >= b:
                    continue
                for s in newadj[a]:
                    if s == b:
                        continue
                    for t in newadj[b]:
                        if t == a or t == s:
                            continue
                        if nd[s] == nd[t]:
                            return False
        return True

    # Start empty, add edges in an order that packs densely.
    # Try all pairs, add if valid.
    order = []
    for u in range(N):
        for v in range(u+1, N):
            order.append((u,v))
    # Try a few heuristics: by index, by random
    import random
    best_edges = []
    best_count = 0

    for attempt in range(200):
        if time.time() > deadline:
            break
        if attempt == 0:
            cand = list(order)
        else:
            cand = list(order)
            random.shuffle(cand)
        adj = [set() for _ in range(N)]
        edges = []
        for (u,v) in cand:
            if time.time() > deadline:
                break
            if u in adj[v]:
                continue
            # quick pre-check: degrees
            du, dv = len(adj[u]), len(adj[v])
            if du+1 == dv+1 and du+1 == dv+1:
                pass
            if valid_after(u, v):
                adj[u].add(v); adj[v].add(u)
                edges.append((u,v))
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
        if best_count >= 90:
            break

    out = []
    for (u,v) in best_edges:
        out.append([int(u), int(v)])
    out.sort()
    return out
