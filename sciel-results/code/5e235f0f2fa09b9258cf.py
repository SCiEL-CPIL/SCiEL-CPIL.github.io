"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix (list of int bitmasks) is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                # Check common neighbors
                common = adj[i] & adj[j]
                if common:
                    return False
    return True

def _is_3_colorable(adj, n):
    """Backtracking 3-colorability check with bitmasks."""
    colors = [0] * n  # 0=unassigned, 1,2,3
    # Order vertices by degree descending for better pruning
    deg = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine forbidden colors
        forbidden = 0
        for u in range(n):
            if adj[v] & (1 << u):
                if colors[u] != 0:
                    forbidden |= (1 << colors[u])
        for c in range(1, 4):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = 0
        return False
    
    return backtrack(0)

def _edge_count(adj, n):
    count = 0
    for i in range(n):
        count += bin(adj[i]).count('1')
    return count // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _can_add_edge(adj, u, v, n):
    """Check if adding edge (u,v) creates a triangle."""
    if adj[u] & (1 << v):
        return False  # already present
    common = adj[u] & adj[v]
    return common == 0

def construct():
    n = 16
    deadline = time.time() + 110.0  # safety margin
    
    # Start with Grötzsch graph (Mycielski of C5) on vertices 0-10
    adj = [0] * n
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        _add_edge(adj, i, (i+1) % 5)
    # Mycielski: vertex 5 connects to 1,4; 6 to 2,0; 7 to 3,1; 8 to 4,2; 9 to 0,3
    for i in range(5):
        _add_edge(adj, i, 5 + ((i+1) % 5))
        _add_edge(adj, (i+1) % 5, 5 + i)
    # Vertex 10 connects to 5,6,7,8,9
    for i in range(5):
        _add_edge(adj, 5+i, 10)
    
    best_adj = adj[:]
    best_edges = _edge_count(adj, n)
    
    # Verify initial
    assert _is_triangle_free(adj, n)
    assert not _is_3_colorable(adj, n)
    
    # Phase 1: Greedily add all possible edges that don't create triangles
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if _can_add_edge(adj, u, v, n):
                    _add_edge(adj, u, v)
                    changed = True
        if time.time() > deadline:
            break
    
    # After greedy, we have a maximal triangle-free graph. Check 4-chromatic.
    if not _is_3_colorable(adj, n):
        best_adj = adj[:]
        best_edges = _edge_count(adj, n)
    
    # Phase 2: Local search - try to swap edges to increase count while maintaining constraints
    # Since we have a maximal triangle-free graph, we can't add more edges directly.
    # But we might be able to restructure. Actually, for a fixed vertex set, a maximal
    # triangle-free graph is unique in terms of edge count? No, different maximal TF graphs
    # can have different edge counts.
    
    # Let's try multiple random restarts with different orderings
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Reset to Grötzsch
        adj = [0] * n
        for i in range(5):
            _add_edge(adj, i, (i+1) % 5)
        for i in range(5):
            _add_edge(adj, i, 5 + ((i+1) % 5))
            _add_edge(adj, (i+1) % 5, 5 + i)
        for i in range(5):
            _add_edge(adj, 5+i, 10)
        
        # Randomize order of edge additions
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if _can_add_edge(adj, u, v, n):
                _add_edge(adj, u, v)
        
        # Check 4-chromatic
        if not _is_3_colorable(adj, n):
            ec = _edge_count(adj, n)
            if ec > best_edges:
                best_adj = adj[:]
                best_edges = ec
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
