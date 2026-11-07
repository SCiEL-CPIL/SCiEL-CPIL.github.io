import time
import random

def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edge_set, n):
    """Check if no simple 3-edge path has endpoints with equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edge_set:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _edge_ok(adj, deg, u, v, n):
    """Check if adding edge (u,v) keeps constraint. Assumes (u,v) not present."""
    # new paths created by adding (u,v)
    # middle: s-u-v-t
    for s in adj[u]:
        if s == v:
            continue
        ds = deg[s]
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[t] == ds:
                return False
    # first: u-v-x-y  (endpoints u, y)
    du = deg[u]
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[y] == du:
                return False
    # last: x-y-v-u (endpoints x, u)
    for x in adj[u]:
        if x == v:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[x] == du:
                return False
    return True

def _havel_hakimi_construct(deg_seq, n):
    """
    Construct a graph with the given degree sequence using Havel-Hakimi.
    Returns a set of edges or None if not graphical.
    """
    # deg_seq is a list of integers
    if len(deg_seq) != n:
        return None
    if sum(deg_seq) % 2 != 0:
        return None
    if any(d < 0 or d > n - 1 for d in deg_seq):
        return None
    
    # Create list of (degree, vertex_id)
    vertices = list(range(n))
    current_deg = deg_seq[:]
    
    edges = set()
    
    # We need to simulate the process.
    # Standard HH: pick max degree, connect to next k vertices, reduce their degrees.
    # This gives a specific graph.
    
    # To allow for multiple solutions or better structure, we can randomize the order of "next k".
    # But for now, let's just get *a* valid graph.
    
    # Copy degrees
    degs = current_deg[:]
    
    for i in range(n):
        if degs[i] < 0:
            return None
        # Find vertex with max degree
        max_idx = -1
        max_deg = -1
        for j in range(n):
            if degs[j] > max_deg:
                max_deg = degs[j]
                max_idx = j
        
        if max_deg == 0:
            # Check if all remaining are 0
            if any(d > 0 for d in degs):
                return None
            break
        if max_deg > n - 1 - i: # Not enough vertices left?
            # Actually, we just need max_deg <= number of non-zero vertices remaining
            non_zero = sum(1 for d in degs if d > 0)
            if max_deg > non_zero - 1:
                return None
        
        # Connect max_idx to the next max_deg vertices with highest degrees
        # Sort indices by degree descending
        indices = [j for j in range(n) if j != max_idx]
        indices.sort(key=lambda j: degs[j], reverse=True)
        
        targets = indices[:max_deg]
        
        for t in targets:
            edges.add((min(max_idx, t), max(max_idx, t)))
            degs[t] -= 1
        
        degs[max_idx] = 0
        
    # Final check
    if any(d != 0 for d in degs):
        return None
        
    # Verify degrees
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != deg_seq[i]:
            return None
            
    return edges

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    n = 15
    rng = random.Random(12345)

    best_edges_set = None
    best_count = -1

    def evaluate_and_track(edge_set):
        nonlocal best_edges_set, best_count
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges_set = set(edge_set)

    # ---------- Initial construction via Havel-Hakimi for high-edge sequences ----------
    # Try a few promising degree sequences
    target_seqs = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14], # 59 edges? No, sum 119.
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14], # Sum 118, 59 edges
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14], # Sum 117. No
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14],  # Sum 116, 58 edges
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14], # Sum 132? No.
    ]
    
    # Add some random high-degree sequences
    for _ in range(10):
        seq = list(range(1, 15))
        # Add 2 extra degrees to make sum even and higher
        # Randomly increment two values
        i, j = rng.sample(range(15), 2)
        seq[i] += 1
        seq[j] += 1
        seq.sort()
        target_seqs.append(seq)

    best_seed_set = set()
    best_seed_count = -1

    for seq in target_seqs:
        if time.time() > deadline - 25:
            break
        edges = _havel_hakimi_construct(seq, n)
        if edges and check_constraint(edges, n):
            if len(edges) > best_seed_count:
                best_seed_count = len(edges)
                best_seed_set = set(edges)

    # Also try random greedy init
    for trial in range(20):
        if time.time() > deadline - 25:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edge_set = set()
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edge_set.add((u, v))
        if len(edge_set) > best_seed_count:
            best_seed_count = len(edge_set)
            best_seed_set = set(edge_set)

    # Start local search from best seed
    edge_set = set(best_seed_set)
    adj = _build_adj(edge_set, n)
    deg = [len(adj[i]) for i in range(n)]
    evaluate_and_track(edge_set)

    # ---------- Local search ----------
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
        deg[u] += 1; deg[v] += 1
        edge_set.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v); adj[v].discard(u)
        deg[u] -= 1; deg[v] -= 1
        edge_set.discard((u, v))

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding missing edges
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            if (u, v) in edge_set:
                continue
            if _edge_ok(adj, deg, u, v, n):
                add_edge(u, v)
                evaluate_and_track(edge_set)
                improved = True
        # Try swaps: remove e1, add e2
        if time.time() < deadline - 5:
            for (a, b) in list(edge_set):
                if time.time() > deadline - 5:
                    break
                for (c, d) in all_pairs:
                    if (c, d) in edge_set:
                        continue
                    # temp remove (a,b)
                    remove_edge(a, b)
                    if _edge_ok(adj, deg, c, d, n):
                        add_edge(c, d)
                        evaluate_and_track(edge_set)
                        improved = True
                    else:
                        # restore
                        add_edge(a, b)

    # Second pass: try 2-swap (remove 2, add 1)
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            improved2 = False
            for (a, b) in list(edge_set):
                if time.time() > deadline - 5:
                    break
                for (c, d) in list(edge_set):
                    if time.time() > deadline - 5:
                        break
                    if (a, b) == (c, d):
                        continue
                    # Check if removing both is valid
                    remove_edge(a, b)
                    remove_edge(c, d)
                    for (e, f) in all_pairs:
                        if (e, f) in edge_set:
                            continue
                        if _edge_ok(adj, deg, e, f, n):
                            add_edge(e, f)
                            evaluate_and_track(edge_set)
                            improved2 = True
                            break
                    if not improved2:
                        add_edge(c, d)
                        add_edge(a, b)
                    else:
                        break
                if improved2:
                    break
            if not improved2:
                break

    result = []
    seen = set()
    for u, v in best_edges_set:
        u = int(u); v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return result
