import time
import random
from itertools import combinations

N = 16
DEADLINE = 110.0
_start_time = None

def _has_q3(adj, u, v):
    """Check if graph with edge (u,v) contains a Q3 that uses edge (u,v)."""
    # A Q3 using edge (u,v): find 6 other vertices such that the 8 induce Q3.
    # Q3: 8 vertices, 12 edges, 3-regular, bipartite (4+4), girth 4.
    # Key: in Q3, each edge is in exactly 2 4-cycles.
    # Given edge (u,v), the other 6 vertices: 
    #   u has 2 other neighbors in Q3, call them u1, u2
    #   v has 2 other neighbors in Q3, call them v1, v2
    #   Then u1-v1, u1-v2, u2-v1, u2-v2 must form a 4-cycle? No...
    # Actually in Q3: u=000, v=001. u's other neighbors: 010, 100. v's other neighbors: 011, 101.
    # Edges: 010-011, 010-110, 100-101, 100-110, 011-111, 101-111, 110-111.
    # So the 6 other vertices are: u1=010, u2=100, v1=011, v2=101, w1=110, w2=111.
    # Structure: u1-v1, u2-v2, u1-w1, u2-w1, v1-w2, v2-w2, w1-w2.
    # This is complex. Let's just enumerate C(14,6) but with fast pruning.
    remaining = [x for x in range(N) if x != u and x != v]
    for combo in combinations(remaining, 6):
        verts = (u, v) + combo
        # Fast check: count edges in induced subgraph
        ec = 0
        for i in range(8):
            vi = verts[i]
            for j in range(i+1, 8):
                if verts[j] in adj[vi]:
                    ec += 1
        if ec != 12:
            continue
        # Check 3-regular
        ok = True
        for i in range(8):
            d = 0
            vi = verts[i]
            for j in range(8):
                if i != j and verts[j] in adj[vi]:
                    d += 1
            if d != 3:
                ok = False
                break
        if not ok:
            continue
        # Check bipartite: try to 2-color
        color = {}
        color[verts[0]] = 0
        queue = [verts[0]]
        bip = True
        while queue:
            x = queue.pop()
            for y in verts:
                if y != x and y in adj[x]:
                    if y not in color:
                        color[y] = 1 - color[x]
                        queue.append(y)
                    elif color[y] == color[x]:
                        bip = False
                        break
            if not bip:
                break
        if bip and len(color) == 8:
            return True
    return False

def _greedy():
    adj = [set() for _ in range(N)]
    edges = []
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_edges)
    for (u, v) in all_edges:
        adj[u].add(v)
        adj[v].add(u)
        if not _has_q3(adj, u, v):
            edges.append([u, v])
        else:
            adj[u].discard(v)
            adj[v].discard(u)
    return edges

def construct():
    global _start_time
    _start_time = time.time()
    best = None
    best_n = 0
    for _ in range(30):
        if time.time() - _start_time > DEADLINE:
            break
        e = _greedy()
        if len(e) > best_n:
            best_n = len(e)
            best = [list(x) for x in e]
    if best is None:
        best = [[a, b] for a in range(3) for b in range(3, 16)]
    return best
