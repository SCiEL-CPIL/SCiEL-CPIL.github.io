"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid(edges, n):
    """Check: no simple 3-edge path s-u-v-t (distinct, s!=t) has deg(s)==deg(t)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    # For each edge (u,v), and each s in adj[u]\{v}, t in adj[v]\{u}, s!=t, s!=v, t!=u (ensured by sets)
    # path s-u-v-t: need s!=t and all distinct. s in adj[u], s!=v; t in adj[v], t!=u; also s!=t, s!=v(already), t!=u(already), u!=v(already edge).
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                # s,u,v,t distinct? s!=u (s in adj[u] so s!=u since no loops), s!=v (checked), s!=t (checked), t!=v (t in adj[v] so t!=v), t!=u (checked), u!=v (edge)
                if deg[s] == deg[t]:
                    return False
    return True

def construct():
    n = 15
    edges = []
    # 5 disjoint triangles: vertices (0,1,2),(3,4,5),...,(12,13,14)
    for base in range(0, n, 3):
        a, b, c = base, base + 1, base + 2
        edges.append([a, b])
        edges.append([b, c])
        edges.append([a, c])
    # Validate (should be vacuously true)
    assert _is_valid(edges, n), "constructed graph invalid"
    # Return as list of [int, int]
    out = []
    for u, v in edges:
        out.append([int(u), int(v)])
    return out
# EVOLVE-BLOCK-END
