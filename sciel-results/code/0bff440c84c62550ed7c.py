import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of a graph on 15 vertices maximizing edge count
    subject to no simple 3-path having endpoints of equal degree.
    Uses local search starting from K_{7,8}.
    """
    n = 15
    # Initial graph: K_{7,8}
    # A: 0..6, B: 7..14
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    def is_valid(edge_set):
        # Compute degrees
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        
        # Check all 3-paths s-u-v-t
        # s, u, v, t must be distinct
        for s in range(n):
            for u in range(n):
                if s == u: continue
                if (s, u) not in edge_set and (u, s) not in edge_set: continue
                for v in range(n):
                    if v == s or v == u: continue
                    if (u, v) not in edge_set and (v, u) not in edge_set: continue
                    for t in range(n):
                        if t == s or t == u or t == v: continue
                        if (v, t) not in edge_set and (t, v) not in edge_set: continue
                        if deg[s] == deg[t]:
                            return False
        return True

    # Helper to check edge existence
    def has_edge(u, v):
        return (u, v) in edges or (v, u) in edges

    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = set(edges)
    best_count = len(edges)
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # Local Search
    while time.time() < deadline:
        # Try to add a random edge
        random_edge = random.choice(all_edges)
        u, v = random_edge
        if not has_edge(u, v):
            # Tentatively add
            edges.add((u, v))
            if is_valid(edges):
                best_edges = set(edges)
                best_count = len(edges)
            else:
                edges.remove((u, v))
        
        # Occasionally try to remove an edge to escape local optima
        if random.random() < 0.1:
            if edges:
                rem_u, rem_v = random.choice(list(edges))
                edges.remove((rem_u, rem_v))
                # Check if removal kept it valid (it might not if degrees change to match)
                if not is_valid(edges):
                    # Revert removal
                    edges.add((rem_u, rem_v))
                else:
                    # Try to add a different edge immediately
                    for _ in range(20):
                        add_u, add_v = random.choice(all_edges)
                        if not has_edge(add_u, add_v):
                            edges.add((add_u, add_v))
                            if is_valid(edges):
                                if len(edges) > best_count:
                                    best_edges = set(edges)
                                    best_count = len(edges)
                                break
                            else:
                                edges.remove((add_u, add_v))

    # Convert to list of lists
    result = []
    for u, v in best_edges:
        result.append([u, v])
    return result
# EVOLVE-BLOCK-END
