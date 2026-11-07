import time
import itertools
import random

# EVOLVE-BLOCK-START

def _validate_and_score(edges, n=15):
    """Return edge count if valid, else -1."""
    if not isinstance(edges, (list, tuple)):
        return -1
    seen = set()
    adj = [[] for _ in range(n)]
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            return -1
        u, v = e
        if type(u) is not int or type(v) is not int:
            return -1
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return -1
        key = (min(u, v), max(u, v))
        if key in seen:
            return -1
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    deg = [len(a) for a in adj]
    # Check: for every edge (u,v), neighbors of u excluding v have distinct degrees,
    # and neighbors of v excluding u have distinct degrees.
    for (u, v) in seen:
        # neighbors of u excluding v
        seen_deg = set()
        for w in adj[u]:
            if w == v:
                continue
            if deg[w] in seen_deg:
                return -1
            seen_deg.add(deg[w])
        # neighbors of v excluding u
        seen_deg = set()
        for w in adj[v]:
            if w == u:
                continue
            if deg[w] in seen_deg:
                return -1
            seen_deg.add(deg[w])
    return len(edges)

def _construct_for_degree_seq(deg_seq, deadline):
    """Try to build a graph with exact degree sequence deg_seq satisfying the constraint."""
    n = len(deg_seq)
    # deg_seq[i] = target degree for vertex i
    # We build edges one by one. Vertex i needs deg_seq[i] edges.
    # Greedy: repeatedly pick an edge (u,v) where u,v still need edges,
    # and adding it doesn't violate the distinct-neighbor constraint.
    
    # For constraint checking during construction:
    # When adding (u,v), we need:
    #   - Among current neighbors of u (excluding v), no two have same target degree
    #   - Among current neighbors of v (excluding u), no two have same target degree
    # But degrees are fixed by target, so we check target degrees of already-connected neighbors.
    
    remaining = list(deg_seq)
    adj = [[] for _ in range(n)]
    edges = []
    
    def can_add(u, v):
        if remaining[u] <= 0 or remaining[v] <= 0:
            return False
        # Check neighbors of u (excluding v): their target degrees must be distinct
        degs_u = set()
        for w in adj[u]:
            if w == v:
                continue
            if deg_seq[w] in degs_u:
                return False
            degs_u.add(deg_seq[w])
        # Check neighbors of v (excluding u)
        degs_v = set()
        for w in adj[v]:
            if w == u:
                continue
            if deg_seq[w] in degs_v:
                return False
            degs_v.add(deg_seq[w])
        return True
    
    # Try to add edges greedily, prioritizing high-remaining vertices
    while True:
        if time.time() > deadline:
            return None
        # Find all valid edges
        valid_edges = []
        for u in range(n):
            if remaining[u] <= 0:
                continue
            for v in range(u+1, n):
                if remaining[v] <= 0:
                    continue
                if can_add(u, v):
                    valid_edges.append((u, v))
        if not valid_edges:
            break
        # Pick the edge connecting the two vertices with highest remaining degree
        valid_edges.sort(key=lambda e: -(remaining[e[0]] + remaining[e[1]]))
        u, v = valid_edges[0]
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
        remaining[u] -= 1
        remaining[v] -= 1
        if all(r <= 0 for r in remaining):
            break
    
    if any(r < 0 for r in remaining):
        return None
    if all(r == 0 for r in remaining):
        return edges
    return None

def _try_all_permutations(deg_targets, deadline):
    """Try multiple permutations of degree targets assigned to vertices."""
    n = len(deg_targets)
    best_edges = None
    best_count = -1
    
    # Try the natural ordering and some random permutations
    permutations_to_try = [list(range(n))]
    # Add some random shuffles
    rng = random.Random(42)
    for _ in range(200):
        p = list(range(n))
        rng.shuffle(p)
        permutations_to_try.append(p)
    
    for perm in permutations_to_try:
        if time.time() > deadline:
            break
        # perm[i] = vertex that gets degree target i
        # So vertex v has target degree = perm.index(v) + 1? No...
        # Let's say deg_seq[v] = perm[v] + 1 (vertex v gets degree perm[v]+1)
        deg_seq = [perm[v] + 1 for v in range(n)]
        # Wait, that's not right either. Let me think again.
        # deg_targets = [1, 2, 3, ..., 15]
        # We want to assign these to vertices 0..14.
        # perm is a permutation of [0..14].
        # Assignment: vertex v gets degree deg_targets[perm[v]]
        deg_seq = [deg_targets[perm[v]] for v in range(n)]
        
        edges = _construct_for_degree_seq(deg_seq, deadline)
        if edges is not None:
            score = _validate_and_score(edges)
            if score > best_count:
                best_count = score
                best_edges = edges
    
    return best_edges, best_count

def _improve(edges, deadline):
    """Try to add more edges to an existing valid graph."""
    if edges is None:
        return None, -1
    n = 15
    adj = [[] for _ in range(n)]
    edge_set = set()
    for e in edges:
        u, v = e
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
    
    improved = True
    current_edges = list(edges)
    
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing edge
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in edge_set:
                    continue
                # Check if adding (u,v) keeps validity
                # Temporarily add and check
                adj[u].append(v)
                adj[v].append(u)
                deg = [len(a) for a in adj]
                valid = True
                # Check all edges
                for (a, b) in edge_set | {(u, v)}:
                    degs_a = set()
                    for w in adj[a]:
                        if w == b:
                            continue
                        if deg[w] in degs_a:
                            valid = False
                            break
                        degs_a.add(deg[w])
                    if not valid:
                        break
                    degs_b = set()
                    for w in adj[b]:
                        if w == a:
                            continue
                        if deg[w] in degs_b:
                            valid = False
                            break
                        degs_b.add(deg[w])
                    if not valid:
                        break
                # Remove
                adj[u].remove(v)
                adj[v].remove(u)
                
                if valid:
                    edge_set.add((u, v))
                    current_edges.append((u, v))
                    improved = True
    
    return current_edges, len(current_edges)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    # Degree targets: 1 through 15 (all distinct)
    deg_targets = list(range(1, 16))
    
    # Try to construct
    best_edges, best_count = _try_all_permutations(deg_targets, deadline)
    
    if best_edges is None or best_count < 0:
        # Fallback: just return the star (always valid)
        return [[0, v] for v in range(1, 15)]
    
    # Try to improve
    if time.time() < deadline:
        improved_edges, improved_count = _improve(best_edges, deadline)
        if improved_count > best_count:
            best_edges = improved_edges
            best_count = improved_count
    
    # Final validation
    final_score = _validate_and_score(best_edges)
    if final_score < 0:
        return [[0, v] for v in range(1, 15)]
    
    # Convert to required format
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END
