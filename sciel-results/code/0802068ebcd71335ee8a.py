"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_cycle_lengths(n, edges):
    """Return set of distinct simple cycle lengths in a graph using DFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycle_lengths = set()
    
    def dfs(start, current, visited, depth):
        if depth > 25:
            return
        for neighbor in adj[current]:
            if neighbor == start and depth >= 4:
                cycle_lengths.add(depth)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, visited, depth + 1)
                visited.remove(neighbor)
    
    for v in range(n):
        dfs(v, v, {v}, 1)
    
    return cycle_lengths


def _validate_and_score(edges, n=12):
    """Validate constraints and return number of distinct cycle lengths, or None if invalid."""
    edge_set = set()
    for u, v in edges:
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
    
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d < 3:
            return None
    
    # Triangle check
    adj_set = set()
    for u, v in edges:
        adj_set.add((min(u,v), max(u,v)))
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in adj_set:
                for w in range(v+1, n):
                    if (u, w) in adj_set and (v, w) in adj_set:
                        return None
    
    cycle_lengths = _get_cycle_lengths(n, edges)
    if not cycle_lengths:
        return None
    
    return len(cycle_lengths)


def _petersen_edges():
    """Return Petersen graph edges on vertices 0-9."""
    # Outer 5-cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = []
    for i in range(5):
        edges.append([i, (i+1) % 5])
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append([inner[i], inner[(i+1) % 5]])
    for i in range(5):
        edges.append([i, inner[i]])
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 12
    best_score = 3
    best_edges = None
    
    # Known good: two K_{3,3}
    edges_k33 = []
    for u in (0, 1, 2):
        for v in (3, 4, 5):
            edges_k33.append([u, v])
    for u in (6, 7, 8):
        for v in (9, 10, 11):
            edges_k33.append([u, v])
    
    score = _validate_and_score(edges_k33, n)
    if score is not None and score < best_score:
        best_score = score
        best_edges = edges_k33
    
    rng = random.Random(42)
    
    # Try Petersen-based constructions
    # Petersen (10 verts) + 2 more vertices
    # Option: Petersen + edge between two new verts won't work (deg 1)
    # Try: Petersen with 2 vertices split into a structure
    # Actually, let's try: use Petersen on 0-9, and add vertices 10,11
    # connected to existing vertices to give them degree 3+
    
    petersen = _petersen_edges()
    # Petersen is 3-regular on 10 vertices
    # Need to add 2 vertices, each with degree >= 3
    # Connect 10 and 11 to 3 vertices each from Petersen, plus edge 10-11
    # This gives 10 and 11 degree 4 each, and the 6 Petersen vertices they connect to get degree 4
    
    for p10 in range(10):
        if time.time() > deadline:
            break
        for p11 in range(10):
            if p11 == p10:
                continue
            # 10 connects to p10 and 2 others, 11 connects to p11 and 2 others
            # Try: 10->p10, 11->p11, and 10-11 edge
            # Plus connect 10 to 2 more, 11 to 2 more
            remaining = [i for i in range(10) if i not in (p10, p11)]
            for a in remaining:
                for b in remaining:
                    if b == a:
                        continue
                    for c in remaining:
                        if c in (a, b):
                            continue
                        edges = [list(e) for e in petersen]
                        edges.append([10, p10])
                        edges.append([10, a])
                        edges.append([10, b])
                        edges.append([11, p11])
                        edges.append([11, c])
                        edges.append([11, 0 if 0 not in (p11, c) else 1])  # placeholder
                        # This is getting complex, let's simplify
                        break
                    break
                break
            break
        break
    
    # Simpler: try random search with better heuristics
    # Focus on sparse graphs close to 3-regular
    
    # Random 3-regular-ish triangle-free graphs
    for trial in range(5000):
        if time.time() > deadline:
            break
        # Generate random graph with edge probability tuned for ~3-regular
        edges = []
        # Use a bipartite-like structure to avoid triangles
        # Split into two sets of 6
        for i in range(6):
            for j in range(6):
                if rng.random() < 0.4:
                    edges.append([i, j + 6])
        
        # Add some intra-set edges if needed for degree
        deg = [0] * 12
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # Add intra-set edges for low-degree vertices
        existing = set((min(u,v), max(u,v)) for u,v in edges)
        for i in range(6):
            if deg[i] < 3:
                for j in range(i+1, 6):
                    if deg[j] < 3 and (i,j) not in existing:
                        edges.append([i, j])
                        deg[i] += 1
                        deg[j] += 1
                        existing.add((i,j))
                        if deg[i] >= 3:
                            break
        for i in range(6, 12):
            if deg[i] < 3:
                for j in range(i+1, 12):
                    if deg[j] < 3 and (i,j) not in existing:
                        edges.append([i, j])
                        deg[i] += 1
                        deg[j] += 1
                        existing.add((i,j))
                        if deg[i] >= 3:
                            break
        
        if min(deg) < 3:
            continue
        s = _validate_and_score(edges, n)
        if s is not None and s < best_score:
            best_score = s
            best_edges = edges
            if best_score == 1:
                break
    
    # Local search: modify best known solution
    if best_edges is not None and time.time() < deadline:
        base = [list(e) for e in best_edges]
        for trial in range(5000):
            if time.time() > deadline:
                break
            edges = [list(e) for e in base]
            # Remove 1-2 edges
            for _ in range(rng.randint(1, 2)):
                if edges:
                    idx = rng.randint(0, len(edges) - 1)
                    edges.pop(idx)
            # Add 1-2 edges
            for _ in range(rng.randint(1, 2)):
                u = rng.randint(0, 11)
                v = rng.randint(0, 11)
                if u != v:
                    key = (min(u,v), max(u,v))
                    if not any(min(a,b) == key[0] and max(a,b) == key[1] for a,b in edges):
                        edges.append([u, v])
            s = _validate_and_score(edges, n)
            if s is not None and s < best_score:
                best_score = s
                best_edges = edges
                base = [list(e) for e in best_edges]
                if best_score == 1:
                    break
    
    if best_edges is None:
        best_edges = edges_k33
    
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
