"""K2,3-free graph on 20 vertices, maximize edges."""
import time
import random
import numpy as np

DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    N = 20
    rng = random.Random(42)

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - elapsed()

    # State: adjacency as set of frozensets for O(1) lookup, plus edge list
    # common[u][v] = number of common neighbors of u and v (for u != v)
    # We track adj as a list of sets

    def make_state(adj_sets):
        """Build common-neighbor counts from adjacency sets."""
        common = [[0] * N for _ in range(N)]
        for u in range(N):
            for v in adj_sets[u]:
                if v > u:
                    # count common neighbors of u and v
                    cnt = len(adj_sets[u] & adj_sets[v])
                    common[u][v] = cnt
                    common[v][u] = cnt
        return common

    def is_k23_free(common):
        """Check no two vertices have >= 3 common neighbors."""
        for u in range(N):
            for v in range(u + 1, N):
                if common[u][v] >= 3:
                    return False
        return True

    def edge_count(adj_sets):
        return sum(len(s) for s in adj_sets) // 2

    def try_add_edge(adj_sets, common, u, v):
        """Try to add edge (u,v). Returns True if added successfully."""
        if u == v or v in adj_sets[u]:
            return False
        # Adding edge (u,v) increases common[w][x] for all w in N(u)\{v} and x in N(v)\{u}
        # Actually, adding (u,v) means u and v become neighbors.
        # For any w != u,v: w is a common neighbor of (u,v) iff w in N(u) and w in N(v).
        # But the K2,3 condition is about common neighbors of PAIRS of vertices.
        # Adding edge (u,v) affects common counts for pairs involving u or v.
        # Specifically, for pair (u, w) where w in N(v): common[u][w] increases by 1 (v becomes common neighbor)
        # For pair (v, w) where w in N(u): common[v][w] increases by 1 (u becomes common neighbor)
        # For pair (w1, w2) where w1 in N(u), w2 in N(v): no direct change to common[w1][w2]
        # Wait, let me think more carefully.
        # common[a][b] = |N(a) ∩ N(b)|
        # Adding edge (u,v): N(u) gains v, N(v) gains u.
        # For pair (u, w) with w != v: N(u) ∩ N(w) gains v iff v in N(w), i.e., w in N(v).
        #   So common[u][w] += 1 for w in N(v) \ {u}
        # For pair (v, w) with w != u: N(v) ∩ N(w) gains u iff u in N(w), i.e., w in N(u).
        #   So common[v][w] += 1 for w in N(u) \ {v}
        # For pair (w1, w2) where w1 != u,v and w2 != u,v: no change.
        # For pair (u, v): common[u][v] = |N(u) ∩ N(v)|, unchanged by adding (u,v) itself.
        
        # Check if any common count would reach 3
        nu = adj_sets[u]
        nv = adj_sets[v]
        
        # Check pairs (u, w) for w in nv
        for w in nv:
            if w == u:
                continue
            if common[u][w] >= 2:  # would become 3
                return False
        # Check pairs (v, w) for w in nu
        for w in nu:
            if w == v:
                continue
            if common[v][w] >= 2:  # would become 3
                return False
        
        # Also need to check: does adding (u,v) create a K2,3 where {u,v} is the "2" part?
        # K2,3 has two vertices (say u,v) each connected to 3 common neighbors.
        # After adding (u,v), the common neighbors of u and v are N(u) ∩ N(v).
        # We need |N(u) ∩ N(v)| <= 2.
        common_uv = len(nu & nv)
        if common_uv >= 3:
            return False
        
        # All checks pass, add the edge
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        for w in nv:
            if w != u:
                common[u][w] += 1
                common[w][u] += 1
        for w in nu:
            if w != v:
                common[v][w] += 1
                common[w][v] += 1
        return True

    def try_remove_edge(adj_sets, common, u, v):
        """Remove edge (u,v). Returns True if removed."""
        if u == v or v not in adj_sets[u]:
            return False
        nu = adj_sets[u]
        nv = adj_sets[v]
        adj_sets[u].remove(v)
        adj_sets[v].remove(u)
        for w in nv:
            if w != u:
                common[u][w] -= 1
                common[w][u] -= 1
        for w in nu:
            if w != v:
                common[v][w] -= 1
                common[w][v] -= 1
        return True

    def greedy_fill(adj_sets, common):
        """Repeatedly add edges in a greedy fashion until no more can be added."""
        changed = True
        while changed:
            changed = False
            # Try all non-edges, pick one that can be added (prefer higher degree balance)
            candidates = []
            for u in range(N):
                for v in range(u + 1, N):
                    if v not in adj_sets[u]:
                        # Check if we can add it
                        nu = adj_sets[u]
                        nv = adj_sets[v]
                        ok = True
                        for w in nv:
                            if w != u and common[u][w] >= 2:
                                ok = False
                                break
                        if ok:
                            for w in nu:
                                if w != v and common[v][w] >= 2:
                                    ok = False
                                    break
                        if ok and len(nu & nv) >= 3:
                            ok = False
                        if ok:
                            # Score: prefer edges between lower-degree vertices for balance
                            score = len(nu) + len(nv)
                            candidates.append((score, u, v))
            if not candidates:
                break
            # Pick a random candidate from the top few for diversity
            candidates.sort()
            top_k = min(5, len(candidates))
            _, u, v = rng.choice(candidates[:top_k])
            if try_add_edge(adj_sets, common, u, v):
                changed = True
        return adj_sets, common

    def simulated_annealing(adj_sets, common, T0=10.0, T1=0.1, cooling=0.999, max_iter=50000):
        """SA with add, remove, and swap moves."""
        current_edges = edge_count(adj_sets)
        best_edges = current_edges
        best_adj = [set(s) for s in adj_sets]
        best_common = [row[:] for row in common]
        
        T = T0
        for it in range(max_iter):
            if time_left() < 5.0:
                break
            
            move_type = rng.random()
            if move_type < 0.3:
                # Try to add a random non-edge
                while True:
                    u = rng.randint(0, N - 1)
                    v = rng.randint(0, N - 1)
                    if u != v and v not in adj_sets[u]:
                        break
                if try_add_edge(adj_sets, common, u, v):
                    new_edges = current_edges + 1
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [set(s) for s in adj_sets]
                        best_common = [row[:] for row in common]
                    current_edges = new_edges
                # else: edge wasn't added, no change
            elif move_type < 0.5:
                # Try to remove a random edge
                if current_edges == 0:
                    continue
                # Pick a random edge
                u = rng.randint(0, N - 1)
                neighbors = list(adj_sets[u])
                if not neighbors:
                    continue
                v = rng.choice(neighbors)
                if try_remove_edge(adj_sets, common, u, v):
                    new_edges = current_edges - 1
                    # Always accept removals in SA (to allow exploration)
                    current_edges = new_edges
            else:
                # Swap: remove one edge, add another
                # Pick a random edge to remove
                u = rng.randint(0, N - 1)
                neighbors = list(adj_sets[u])
                if not neighbors:
                    continue
                v = rng.choice(neighbors)
                # Temporarily remove
                if not try_remove_edge(adj_sets, common, u, v):
                    continue
                current_edges -= 1
                # Try to add a random non-edge
                added = False
                for _ in range(10):
                    u2 = rng.randint(0, N - 1)
                    v2 = rng.randint(0, N - 1)
                    if u2 != v2 and v2 not in adj_sets[u2]:
                        if try_add_edge(adj_sets, common, u2, v2):
                            added = True
                            current_edges += 1
                            break
                if not added:
                    # Restore the removed edge
                    try_add_edge(adj_sets, common, u, v)
                    current_edges += 1
                    continue
                # Metropolis acceptance
                delta = current_edges - (best_edges if current_edges > best_edges else current_edges)
                # Actually, accept if it doesn't decrease edge count, or with probability exp(-delta/T)
                # For simplicity, accept if edge count didn't decrease
                if current_edges >= best_edges:
                    pass  # keep
                elif rng.random() < np.exp(-1.0 / max(T, 1e-10)):
                    pass  # accept worsening
                else:
                    # Reject: undo the swap
                    try_remove_edge(adj_sets, common, u2, v2)
                    current_edges -= 1
                    try_add_edge(adj_sets, common, u, v)
                    current_edges += 1
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj_sets]
                    best_common = [row[:] for row in common]
            
            T *= cooling
        
        return best_adj, best_common, best_edges

    # Multi-strategy initialization
    best_edges = 0
    best_adj = None
    best_common = None

    strategies = []
    
    # Strategy 1: Random dense initialization
    for density in [0.3, 0.4, 0.5]:
        adj = [set() for _ in range(N)]
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < density:
                    adj[u].add(v)
                    adj[v].add(u)
        strategies.append(adj)
    
    # Strategy 2: Circulant-like
    adj = [set() for _ in range(N)]
    shifts = [1, 2, 4, 5, 7]
    for u in range(N):
        for s in shifts:
            v = (u + s) % N
            if v > u:
                adj[u].add(v)
                adj[v].add(u)
    strategies.append(adj)
    
    # Strategy 3: Sparse random
    adj = [set() for _ in range(N)]
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < 0.2:
                adj[u].add(v)
                adj[v].add(u)
    strategies.append(adj)

    for init_adj in strategies:
        if time_left() < 10.0:
            break
        
        # Greedy fill
        adj = [set(s) for s in init_adj]
        common = make_state(adj)
        adj, common = greedy_fill(adj, common)
        edges = edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = [set(s) for s in adj]
            best_common = [row[:] for row in common]
        
        # SA
        if time_left() > 15.0:
            sa_adj, sa_common, sa_edges = simulated_annealing(adj, common)
            if sa_edges > best_edges:
                best_edges = sa_edges
                best_adj = [set(s) for s in sa_adj]
                best_common = [row[:] for row in sa_common]
        
        # Final greedy fill after SA
        if time_left() > 5.0:
            adj, common = greedy_fill([set(s) for s in best_adj], [row[:] for row in best_common])
            edges = edge_count(adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = [set(s) for s in adj]
                best_common = [row[:] for row in common]

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if v > u:
                edges.append([u, v])
    
    return edges
