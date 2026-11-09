"""Line-graph MIS approach: maximize triangle-free edges on 16 vertices with chi>=4."""

import time
import random
import sys
from itertools import combinations

_DEADLINE = None

def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        return True
    return time.time() < _DEADLINE

def _is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Backtracking 3-colorability check. Returns True if 3-colorable."""
    color = [-1] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = [False, False, False]
        for nb in adj[v]:
            if pos_in_order[nb] < idx:
                forbidden[color[nb]] = True
        # Try most constrained first
        for c in range(3):
            if not forbidden[c]:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) preserves triangle-free property."""
    # For triangle-free: no common neighbor of u and v
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def _greedy_independent_set(candidates, conflict, remaining_mask):
    """Greedy: pick vertices with highest degree in candidate subgraph."""
    chosen = []
    mask = remaining_mask
    while mask and _time_ok():
        # Find vertex with max degree in current candidate set
        best_v = -1
        best_deg = -1
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Count neighbors in mask
            deg = (conflict[v] & mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= lsb
        if best_v < 0:
            break
        chosen.append(best_v)
        mask ^= (1 << best_v)
        mask &= ~conflict[best_v]
    return chosen, mask

def _max_independent_set_line_graph(n, time_budget=100):
    """
    Find maximum independent set in line graph of K_n.
    Each node in line graph = edge (u,v) with u<v in K_n.
    Two nodes adjacent if edges share a vertex (would form triangle if both chosen).
    Independent set = set of edges no two share a vertex = matching? No wait...
    
    Actually: triangle-free means no three vertices all mutually connected.
    Edge e1=(a,b), e2=(c,d), e3=(e,f) form triangle iff {a,b}∪{c,d}∪{e,f} = {x,y,z} with all pairs present.
    
    Two edges conflict if they share a vertex AND there's a path of length 2 between the other endpoints.
    Actually simpler: edges (u,v) and (w,x) conflict if they share a vertex, say u=w, and v is adjacent to x in the graph.
    But we're building the graph, so we need: if we include edges (u,v) and (u,w), then we CANNOT include (v,w).
    
    So the constraint is: for any vertex u, the neighbors of u in our graph must form an independent set.
    This is equivalent to: the set of all edges must be such that for every triple (u,v,w), not all three edges (u,v),(v,w),(u,w) are present.
    
    Line graph formulation: nodes = potential edges (u,v) u<v.
    Two potential edges e1=(a,b), e2=(c,d) are adjacent in line graph if:
    - They share a vertex: a=c or a=d or b=c or b=d
    - AND the other two vertices are connected by a third edge
    Wait no. Let me think again.
    
    Actually the correct formulation:
    We want to select a set S of edges from K_n such that S is triangle-free.
    Triangle-free means: for any (u,v), (v,w), (u,w), not all three are in S.
    
    This is a 3-uniform hypergraph independent set problem, not a graph one.
    But we can use a different approach: branch and bound directly on edges.
    
    Let me use a simpler branch-and-bound:
    - Order all C(16,2)=120 possible edges
    - At each step, decide to include or exclude the next edge
    - Prune if including would create a triangle
    - Upper bound: current count + remaining edges that don't conflict
    """
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    m_total = len(all_edges)
    
    # For each edge, precompute which other edges would form a triangle with it
    # Edge (u,v) conflicts with (u,w) and (v,w) for any w != u,v
    # So if we have (u,v) and (u,w), we cannot have (v,w)
    
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # conflict[i] = set of edge indices j such that including both i and j prevents some third edge
    # Actually: including edges i and j "blocks" certain third edges
    # Let's think differently:
    # State: set of included edges E
    # Can include edge (u,v) if: for all w, not (both (u,w) and (v,w) in E)
    
    # For efficiency, track for each pair of vertices whether they're connected
    # And for each vertex, its neighbors
    
    best_edges = []
    best_count = 0
    
    # Use recursive backtracking with good ordering
    # Order edges by some heuristic
    
    # Precompute for each edge, the set of edges that would be "blocked" if this edge is added
    # Edge (u,v) blocks edge (w,x) if {u,v,w,x} contains a triangle with (u,v)
    # Specifically, (u,v) and (u,w) block (v,w)
    # (u,v) and (v,w) block (u,w)
    # (u,v) and (w,x) with w,x not u,v: no direct block, but if w=u then it's (u,v) and (u,x) blocking (v,x)
    
    # Let's just do straightforward backtracking with pruning
    
    sys.setrecursionlimit(10000)
    
    # Order: try edges in an order that helps prune
    # Maybe sort by some criterion
    
    def backtrack(idx, included, adj_sets, count):
        nonlocal best_edges, best_count
        
        if not _time_ok():
            return
        
        if count > best_count:
            best_count = count
            best_edges = list(included)
        
        if idx == m_total:
            return
        
        # Upper bound: count + remaining edges
        if count + (m_total - idx) <= best_count:
            return
        
        u, v = all_edges[idx]
        
        # Option 1: Include edge (u,v)
        # Check if it creates a triangle
        can_include = True
        for w in range(n):
            if w != u and w != v:
                if (u, w) in adj_sets and (v, w) in adj_sets:
                    can_include = False
                    break
        
        if can_include:
            included.append((u, v))
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            backtrack(idx + 1, included, adj_sets, count + 1)
            included.pop()
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
        
        # Option 2: Exclude edge (u,v)
        backtrack(idx + 1, included, adj_sets, count)
    
    # This naive approach is too slow for 120 edges.
    # Let's use a better approach: greedy + local search
    
    # Greedy construction
    adj = [set() for _ in range(n)]
    
    # Start with Grötzsch graph as seed (known 4-chromatic triangle-free on 11 vertices)
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    # C5
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    
    # Mycielski
    for i in range(5):
        left = (i - 1) % 5
        right = (i + 1) % 5
        add_edge(5 + i, left)
        add_edge(5 + i, right)
    
    for i in range(5):
        add_edge(10, 5 + i)
    
    # Now greedily add more edges
    # For each potential edge not yet present, try to add it if it doesn't create a triangle
    
    # Sort potential edges by some heuristic
    potential = []
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                potential.append((u, v))
    
    # Sort by: prefer edges between vertices with low common neighborhood
    def edge_score(e):
        u, v = e
        # Number of w such that w is neighbor of both u and v
        common = len(adj[u] & adj[v])
        return common
    
    potential.sort(key=edge_score)
    
    for u, v in potential:
        if not _time_ok():
            break
        # Check if adding (u,v) creates a triangle
        can_add = True
        for w in adj[u]:
            if w in adj[v]:
                can_add = False
                break
        if can_add:
            add_edge(u, v)
    
    # Local search: try to improve by swapping edges
    current_edges = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                current_edges.add((u, v))
    
    improved = True
    while improved and _time_ok():
        improved = False
        # Try removing one edge and adding another
        for e1 in list(current_edges):
            u1, v1 = e1
            # Temporarily remove
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            current_edges.discard(e1)
            
            # Try to add as many edges as possible
            added = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        # Check if can add
                        can_add = True
                        for w in adj[u]:
                            if w in adj[v]:
                                can_add = False
                                break
                        if can_add:
                            adj[u].add(v)
                            adj[v].add(u)
                            current_edges.add((u, v))
                            added.append((u, v))
            
            if len(added) > 1:
                improved = True
            else:
                # Revert
                for u, v in added:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    current_edges.discard((u, v))
                adj[u1].add(v1)
                adj[v1].add(u1)
                current_edges.add(e1)
    
    # Convert to edge list
    edges = [[u, v] for (u, v) in current_edges]
    
    return edges

def _build_grotzsch(n):
    """Build Grötzsch graph on first 11 vertices, rest isolated."""
    adj = [set() for _ in range(n)]
    # C5
    for i in range(5):
        adj[i].add((i + 1) % 5)
        adj[(i + 1) % 5].add(i)
    # Mycielski
    for i in range(5):
        left = (i - 1) % 5
        right = (i + 1) % 5
        adj[5 + i].add(left)
        adj[left].add(5 + i)
        adj[5 + i].add(right)
        adj[right].add(5 + i)
    for i in range(5):
        adj[10].add(5 + i)
        adj[5 + i].add(10)
    return adj

def _greedy_add_edges(adj, n, potential, order=None):
    """Greedily add edges from potential list that don't create triangles."""
    if order is None:
        order = potential
    for u, v in order:
        if v in adj[u]:
            continue
        if _can_add_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)

def _local_search(adj, n, max_iter=200):
    """Local search: try removing one edge and adding multiple others."""
    best_count = sum(len(adj[u]) for u in range(n)) // 2
    best_adj = [set(s) for s in adj]
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Pick a random edge to remove
        all_edges = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    all_edges.append((u, v))
        
        if not all_edges:
            break
        
        random.shuffle(all_edges)
        
        for u1, v1 in all_edges[:10]:  # Try up to 10 removals
            if not _time_ok():
                break
            
            # Remove edge
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            
            # Try to add as many edges as possible
            added = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u] and _can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added.append((u, v))
            
            new_count = sum(len(adj[u]) for u in range(n)) // 2
            
            if new_count > best_count:
                best_count = new_count
                best_adj = [set(s) for s in adj]
                # Keep the improvement, continue searching
            else:
                # Revert
                for u, v in added:
                    adj[u].discard(v)
                    adj[v].discard(u)
                adj[u1].add(v1)
                adj[v1].add(u1)
    
    # Restore best
    for u in range(n):
        adj[u] = set(best_adj[u])
    
    return best_count

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 110  # safety margin
    
    random.seed(42)
    n = 16
    
    # All potential edges
    all_potential = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = None
    best_count = 0
    
    # Strategy 1: Multiple random restarts with greedy + local search
    for trial in range(200):
        if not _time_ok():
            break
        
        # Start from Grötzsch
        adj = _build_grotzsch(n)
        
        # Shuffle potential edges
        potential = list(all_potential)
        random.shuffle(potential)
        
        # Greedy add
        _greedy_add_edges(adj, n, potential)
        
        # Check 3-colorability
        if _is_3_colorable(adj, n):
            continue
        
        # Count edges
        count = sum(len(adj[u]) for u in range(n)) // 2
        
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
        
        # Local search to improve
        new_count = _local_search(adj, n, max_iter=50)
        
        if new_count > best_count:
            best_count = new_count
            best_edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
        
        # Early exit if we hit theoretical max for triangle-free on 16 vertices
        # Mantel's theorem: max edges in triangle-free on n vertices = n²/4 = 64
        if best_count >= 64:
            break
    
    # Strategy 2: If we haven't found good solution, try different approach
    if best_count < 30:
        # Try starting from different seeds
        for seed_offset in range(10):
            if not _time_ok():
                break
            
            random.seed(42 + seed_offset * 1000)
            adj = _build_grotzsch(n)
            
            # Add some random edges first
            random.shuffle(all_potential)
            for u, v in all_potential[:20]:
                if _can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            if not _is_3_colorable(adj, n):
                count = sum(len(adj[u]) for u in range(n)) // 2
                if count > best_count:
                    best_count = count
                    best_edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
                
                new_count = _local_search(adj, n, max_iter=30)
                if new_count > best_count:
                    best_count = new_count
                    best_edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
    
    # Final validation
    if best_edges is not None:
        # Verify triangle-free
        adj_final = [set() for _ in range(n)]
        for u, v in best_edges:
            adj_final[u].add(v)
            adj_final[v].add(u)
        
        if _is_triangle_free(adj_final, n) and not _is_3_colorable(adj_final, n):
            return best_edges
    
    # Fallback: Grötzsch + isolated vertices
    adj = _build_grotzsch(n)
    edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
    return edges
