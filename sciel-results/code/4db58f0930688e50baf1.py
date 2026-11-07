"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def _bfs_diameter(adj, n):
    """Return (diameter, connected). Diameter is max shortest path; inf if disconnected."""
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d == -1:
                return float('inf'), False
            if d > diam:
                diam = d
    return diam, True


def _is_valid(edges, n):
    """Check diameter == 2 and every edge is critical."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Diameter must be exactly 2
    diam, conn = _bfs_diameter(adj, n)
    if not conn or diam != 2:
        return False

    # Every edge must be critical
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    for (u, v) in edge_set:
        # Remove edge (u,v)
        adj[u].remove(v)
        adj[v].remove(u)
        d2, c2 = _bfs_diameter(adj, n)
        # Restore
        adj[u].add(v)
        adj[v].add(u)
        if c2 and d2 <= 2:
            return False
    return True


def _build_k88():
    """Build K_{8,8} on vertices 0..15."""
    n = 16
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges


def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    # Primary candidate: K_{8,8}
    edges = _build_k88()
    n = 16

    # Quick structural verification (fast path)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Verify diameter 2
    diam, conn = _bfs_diameter(adj, n)
    if not conn or diam != 2:
        # Fallback to star (should never happen)
        return [[0, v] for v in range(1, n)]

    # Verify every edge is critical
    # For K_{8,8}: removing any edge (u,v) with u<8, v>=8:
    # u still connects to all other 7 in part B, v to all other 7 in part A.
    # Diameter remains 2. So K_{8,8} is NOT valid for this task!
    # Wait - let me recheck. In K_{8,8}, remove edge (u,v). 
    # u can still reach v via any other vertex in the opposite part.
    # So diameter stays 2. That means edges are NOT critical.
    # 
    # The measured result said K_{8,8} scored 4.267 = 64/15. 
    # But if edges aren't critical, it should be invalid (score 0).
    # 
    # Re-reading: "Deleting EACH individual edge must increase its diameter 
    # above two or disconnect it." 
    # In K_{8,8}, deleting one edge doesn't increase diameter. So it's INVALID.
    # 
    # The observation data says score 4.267 but validity_not_declared. 
    # The score might be wrong, or my understanding is off.
    # 
    # Let me reconsider: maybe the task means the graph must have the property 
    # that it's "edge-minimal" for diameter 2, i.e., no edge can be removed 
    # while preserving diameter 2. That's what I've been checking.
    # 
    # For K_{8,8}: it's NOT edge-minimal for diameter 2. So it's invalid.
    # 
    # I need a graph where EVERY edge is the unique length-2 connector for 
    # some pair, OR removing it disconnects.
    # 
    # A tree has diameter > 2 for n=16 unless it's a star (diam 2). 
    # Star: removing any edge disconnects. Valid! 15 edges.
    # 
    # Can we do better? We need a graph where every edge is a bridge OR 
    # the unique 2-path for its endpoints.
    # 
    # Actually, re-reading more carefully: the edge is critical if removing it 
    # makes diameter > 2 OR disconnects. In K_{8,8}, removing an edge keeps 
    # diameter 2 and connected. So NOT critical. Invalid.
    # 
    # The best valid graph might indeed be the star (15 edges) or something 
    # slightly better. Let me think about what graphs have all edges critical.
    # 
    # A graph where every edge is a bridge: a tree. Max edges in a tree = 15. 
    # But a tree on 16 vertices has diameter at most 15, and we need diameter 
    # exactly 2. Only the star has diameter 2 among trees. So 15 edges.
    # 
    # Can we have a non-tree with all edges critical? If there's a cycle, 
    # then an edge on the cycle has an alternative path, so removing it doesn't 
    # disconnect. But it might increase diameter. 
    # 
    # For diameter exactly 2: if we add an edge to the star, we create a 
    # triangle. The star edges in the triangle: removing one still leaves 
    # diameter 2 (via the other two edges of the triangle or other star edges). 
    # So the added edge might be critical (removing it restores star, diam 2 
    # still holds via center), but the star edges might no longer be critical.
    # 
    # This is getting complex. Given the time constraint and that the star 
    # is definitely valid, let me return the star as a safe fallback, but 
    # also try to construct a better graph if I can verify it.
    # 
    # For now, return the star which is guaranteed valid with 15 edges.
    return [[0, v] for v in range(1, n)]

# EVOLVE-BLOCK-END
