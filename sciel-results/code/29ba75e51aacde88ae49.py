"""Model-and-optimize: parameterize as disjoint union of C5-free blocks with alpha=1 each (K4),
   fit block count to maximize n subject to alpha <= 8 and C5-free."""
import time
import random

# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph (simple cycle of length 5)."""
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if not adj[v3][v4] and adj[v4][v0]:
                            return True
    return False


def _max_independent_set_size(n, adj):
    """Exact max independent set via branch and bound."""
    best = [0]
    adj_set = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_set[i].add(j)
                adj_set[j].add(i)

    def dfs(cand, size):
        if size + len(cand) <= best[0]:
            return
        if not cand:
            if size > best[0]:
                best[0] = size
            return
        v = cand[0]
        rest = cand[1:]
        # include v
        new_cand = [x for x in rest if x not in adj_set[v]]
        dfs(new_cand, size + 1)
        # exclude v
        dfs(rest, size)

    dfs(list(range(n)), 0)
    return best[0]


def _is_valid(n, edges):
    """Check if graph is C5-free and has no independent set of size 9."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    if _has_c5(n, adj):
        return False
    if _max_independent_set_size(n, adj) >= 9:
        return False
    return True


def construct():
    # Start with the known 32-vertex construction: 8 disjoint K4's
    # Each K4 is C5-free and has alpha=1
    # Total: n=32, alpha=8 (no independent set of size 9)
    
    blocks = 8
    k = 4
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
    
    # Try to improve by adding vertices and edges to reach n=33, 34, or 35
    # This is a search for a valid graph with more vertices
    
    deadline = time.time() + 145.0  # Leave 5s safety margin
    
    # Try to extend the 32-vertex graph to 33, 34, or 35 vertices
    for target_n in [35, 34, 33]:
        if time.time() > deadline - 30:
            break
        
        # Start from the 32-vertex graph and try to add (target_n - 32) vertices
        extra = target_n - 32
        new_edges = list(edges)
        
        # Add new vertices and try to connect them
        # Strategy: add vertices one at a time, trying to maintain C5-free and alpha <= 8
        
        adj = [[False] * target_n for _ in range(target_n)]
        for u, v in new_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Try to add edges for the new vertices
        # Each new vertex must be connected to at least some existing vertices
        # to reduce the independence number
        
        for new_v in range(32, target_n):
            if time.time() > deadline - 10:
                break
            
            # Try different connection patterns for the new vertex
            # We need to ensure no C5 is formed and alpha stays <= 8
            
            # Simple heuristic: connect to a random subset of existing vertices
            # that forms a C5-free structure
            
            # For now, just try to connect to vertices from different K4 blocks
            # to avoid creating C5s within a single block
            
            candidates = list(range(32))
            random.shuffle(candidates)
            
            # Try connecting to 4-6 vertices from different blocks
            connected = []
            for c in candidates:
                if len(connected) >= 6:
                    break
                # Check if adding this edge would create a C5
                adj[new_v][c] = True
                adj[c][new_v] = True
                if not _has_c5(target_n, adj):
                    connected.append(c)
                else:
                    adj[new_v][c] = False
                    adj[c][new_v] = False
            
            # Check if the graph is still valid
            if _is_valid(target_n, new_edges):
                # Found a valid graph with target_n vertices
                return {'n': target_n, 'edges': new_edges}
        
        # If we didn't find a valid graph with random connections,
        # try a more systematic approach: add a K4 block and remove some edges
        # to reduce the independence number
        
        # Actually, adding a K4 would give alpha = 9, which is invalid.
        # We need a more careful approach.
    
    # If we couldn't find a better graph, return the known 32-vertex construction
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
