"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 125  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as list of sets for fast lookup
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    # Track best state
    best_adj = [set(s) for s in adj]
    best_count = len([1 for u in range(n) for v in adj[u]]) // 2
    
    def has_c6(u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through new edge u-v means there's a path of length 5 from u to v
        in the current graph (not using edge u-v since it doesn't exist yet).
        Equivalently, there exist vertices a,b,c,d,e such that
        u-a-b-c-d-e-v is a path of length 6 (7 vertices, 6 edges),
        and all consecutive pairs are edges, and all 7 vertices are distinct.
        
        Actually, adding edge (u,v) creates a C6 iff there is a path of length 5
        from u to v in the current graph with all intermediate vertices distinct
        from u and v.
        """
        # BFS/DFS from u to find path of length exactly 5 to v
        # Path: u -> x1 -> x2 -> x3 -> x4 -> v (5 edges, 4 intermediate vertices)
        # All of x1..x4 must be distinct and not equal to u or v
        
        # Use iterative deepening or just BFS up to depth 5
        # State: (current_node, depth, visited_set)
        # But visited_set makes it expensive. Let's use a different approach.
        
        # For each pair (a, e) where a is neighbor of u and e is neighbor of v,
        # check if there's a path of length 3 from a to e avoiding u, v, and with
        # all intermediate vertices distinct from a, e, u, v.
        
        # Path: u-a-b-c-e-v means a-b-c-e is path of length 3
        # b and c are intermediate, must be distinct from a, e, u, v, and from each other
        
        neighbors_u = adj[u] - {v}
        neighbors_v = adj[v] - {u}
        
        for a in neighbors_u:
            if a == v:
                continue
            for e in neighbors_v:
                if e == u or e == a:
                    continue
                # Need path a -> b -> c -> e of length 3
                # b in adj[a], c in adj[b] ∩ adj[e], b != c, b != e, c != a, b != u, b != v, c != u, c != v
                for b in adj[a]:
                    if b == u or b == v or b == a or b == e:
                        continue
                    for c in adj[b]:
                        if c == u or c == v or c == a or c == b or c == e:
                            continue
                        if e in adj[c]:
                            return True
        return False
    
    # Greedy edge addition with multiple passes
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    

    
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(all_pairs)
        added_any = False
        
        for (u, v) in all_pairs:
            if time.time() > deadline - 3:
                break
            if v in adj[u]:
                continue
            if not has_c6(u, v):
                add_edge(u, v)
                added_any = True
        
        current_count = len([1 for u in range(n) for v in adj[u]]) // 2
        if current_count > best_count:
            best_count = current_count
            best_adj = [set(s) for s in adj]
        
        if not added_any:
            # Try perturbation: remove a random edge and try to re-add different ones
            if time.time() > deadline - 3:
                break
            # Pick a random edge to remove
            all_current_edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if all_current_edges:
                ru, rv = random.choice(all_current_edges)
                adj[ru].discard(rv)
                adj[rv].discard(ru)
                # Now try to add edges again
                random.shuffle(all_pairs)
                for (u, v) in all_pairs:
                    if time.time() > deadline - 3:
                        break
                    if v in adj[u]:
                        continue
                    if not has_c6(u, v):
                        add_edge(u, v)
                current_count = len([1 for u in range(n) for v in adj[u]]) // 2
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [set(s) for s in adj]
            else:
                break
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
