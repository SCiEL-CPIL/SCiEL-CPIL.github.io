import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)
    
    def _validate(edges):
        """Verify no two vertices have 3+ common neighbors."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        for u in range(n):
            for v in range(n):
                if u >= v:
                    continue
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                if cnt >= 3:
                    return False
        return True
    
    def _build_state(edges):
        """Build adjacency and common-neighbor counts from edge list."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                common[u][v] = cnt
        return adj, common
    
    def _can_add(adj, common, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if adj[u][v]:
            return False
        if common[u][v] >= 3:
            return False
        # For each neighbor w of u (w != v): common[w][v] must be < 2
        for w in range(n):
            if w != v and adj[u][w]:
                if common[w][v] >= 2:
                    return False
        # For each neighbor w of v (w != u): common[u][w] must be < 2
        for w in range(n):
            if w != u and adj[v][w]:
                if common[u][w] >= 2:
                    return False
        return True
    
    def _do_add(adj, common, u, v):
        """Add edge (u,v) and update common counts."""
        adj[u][v] = adj[v][u] = True
        for w in range(n):
            if w != v and adj[u][w]:
                common[w][v] += 1
                common[v][w] += 1
        for w in range(n):
            if w != u and adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
    
    def _do_remove(adj, common, u, v):
        """Remove edge (u,v) and update common counts."""
        adj[u][v] = adj[v][u] = False
        for w in range(n):
            if w != v and adj[u][w]:
                common[w][v] -= 1
                common[v][w] -= 1
        for w in range(n):
            if w != u and adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
    
    def _greedy_fill(adj, common, edges, seed, max_rounds=50):
        """Repeatedly add any valid edge until none remain. Returns updated edges."""
        rng = random.Random(seed)
        for _ in range(max_rounds):
            added = False
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            for u, v in shuffled:
                if _can_add(adj, common, u, v):
                    _do_add(adj, common, u, v)
                    edges.append([u, v])
                    added = True
            if not added:
                break
        return edges
    
    def _local_search(edges, seed, time_budget=5.0):
        """
        Local search: try adding edges; when stuck, remove a random edge to escape.
        Uses a simulated-annealing-like temperature for accepting worse moves.
        Returns best edges found.
        """
        rng = random.Random(seed)
        adj, common = _build_state(edges)
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        best_edges = [e[:] for e in current_edges]
        best_count = current_count
        
        # Temperature schedule
        t0 = time.time()
        max_iter = 20000
        no_improve = 0
        patience = 3000
        
        for it in range(max_iter):
            if time.time() - t0 > time_budget:
                break
            if no_improve > patience:
                break
            
            # Try to add a valid edge
            added = False
            for _ in range(30):
                u, v = rng.choice(all_edges)
                if _can_add(adj, common, u, v):
                    _do_add(adj, common, u, v)
                    current_edges.append([u, v])
                    current_count += 1
                    added = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = [e[:] for e in current_edges]
                        no_improve = 0
                    break
            
            if not added:
                # Stuck: try removing a random edge to enable new additions
                if current_edges:
                    idx = rng.randrange(len(current_edges))
                    u, v = current_edges[idx]
                    _do_remove(adj, common, u, v)
                    current_edges.pop(idx)
                    current_count -= 1
                    no_improve += 1
                    
                    # After removal, try to add edges again
                    for _ in range(20):
                        uu, vv = rng.choice(all_edges)
                        if _can_add(adj, common, uu, vv):
                            _do_add(adj, common, uu, vv)
                            current_edges.append([uu, vv])
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_edges = [e[:] for e in current_edges]
                                no_improve = 0
                            break
        
        return best_edges, best_count
    
    def _two_opt_swap(edges, seed, time_budget=3.0):
        """
        Try removing one edge and adding two different edges (net +1).
        This is a stronger move than single add/remove.
        """
        rng = random.Random(seed)
        adj, common = _build_state(edges)
        current_edges = [e[:] for e in edges]
        current_count = len(current_edges)
        best_edges = [e[:] for e in current_edges]
        best_count = current_count
        
        t0 = time.time()
        max_iter = 5000
        no_improve = 0
        patience = 2000
        
        for it in range(max_iter):
            if time.time() - t0 > time_budget:
                break
            if no_improve > patience:
                break
            
            # Try: remove edge e1, add edge e2 and e3
            if len(current_edges) < 2:
                break
            
            idx1 = rng.randrange(len(current_edges))
            u1, v1 = current_edges[idx1]
            
            # Try removing e1 and adding two new edges
            e2 = rng.choice(all_edges)
            e3 = rng.choice(all_edges)
            
            # Check if e2 and e3 are different from e1 and each other
            if (e2[0] == u1 and e2[1] == v1) or (e2[0] == v1 and e2[1] == u1):
                continue
            if (e3[0] == u1 and e3[1] == v1) or (e3[0] == v1 and e3[1] == u1):
                continue
            if (e3[0] == e2[0] and e3[1] == e2[1]) or (e3[0] == e2[1] and e3[1] == e2[0]):
                continue
            if adj[e2[0]][e2[1]] or adj[e3[0]][e3[1]]:
                continue
            
            # Tentatively remove e1
            _do_remove(adj, common, u1, v1)
            
            # Try to add e2
            if _can_add(adj, common, e2[0], e2[1]):
                _do_add(adj, common, e2[0], e2[1])
                # Try to add e3
                if _can_add(adj, common, e3[0], e3[1]):
                    _do_add(adj, common, e3[0], e3[1])
                    current_edges.pop(idx1)
                    current_edges.append([e2[0], e2[1]])
                    current_edges.append([e3[0], e3[1]])
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = [e[:] for e in current_edges]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Roll back e2
                    _do_remove(adj, common, e2[0], e2[1])
                    no_improve += 1
            else:
                no_improve += 1
            
            # Roll back e1
            _do_add(adj, common, u1, v1)
            current_edges.insert(idx1, [u1, v1])
        
        return best_edges, best_count
    
    def _greedy_construct(seed):
        """Greedy construction with random edge ordering."""
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        edges = []
        
        order = all_edges[:]
        rng.shuffle(order)
        
        for u, v in order:
            if _can_add(adj, common, u, v):
                _do_add(adj, common, u, v)
                edges.append([u, v])
        
        return edges
    
    def _structured_init(seed):
        """Create a structured initial graph (circulant-like)."""
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        # Circulant: connect each vertex to its k nearest neighbors
        k = rng.randint(3, 6)
        for i in range(n):
            for d in range(1, k + 1):
                j = (i + d) % n
                if i < j and not adj[i][j]:
                    adj[i][j] = adj[j][i] = True
                    edges.append([i, j])
        
        # Add some random edges
        for _ in range(rng.randint(5, 15)):
            u, v = rng.choice(all_edges)
            if not adj[u][v]:
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
        
        return edges
    
    # Main search
    best_edges = []
    best_count = 0
    
    # Phase 1: Greedy constructions
    greedy_results = []
    for seed in range(100):
        if time.time() > deadline - 60.0:
            break
        result = _greedy_construct(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result[:]
        greedy_results.append((len(result), result))
    
    # Phase 2: Structured initializations + greedy fill
    for seed in range(30):
        if time.time() > deadline - 40.0:
            break
        init_edges = _structured_init(seed)
        adj, common = _build_state(init_edges)
        edges = _greedy_fill(adj, common, init_edges, seed=5000 + seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        greedy_results.append((len(edges), edges))
    
    # Sort by quality
    greedy_results.sort(key=lambda x: -x[0])
    
    # Phase 3: Local search from top results
    for i in range(min(10, len(greedy_results))):
        if time.time() > deadline - 20.0:
            break
        _, start_edges = greedy_results[i]
        ls_edges, ls_count = _local_search(start_edges, seed=10000 + i, time_budget=3.0)
        if ls_count > best_count:
            best_count = ls_count
            best_edges = ls_edges[:]
    
    # Phase 4: Two-opt swap from best
    if time.time() < deadline - 10.0:
        swap_edges, swap_count = _two_opt_swap(best_edges, seed=20000, time_budget=5.0)
        if swap_count > best_count:
            best_count = swap_count
            best_edges = swap_edges[:]
    
    # Phase 5: Final polish - local search from best with more time
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        final_edges, final_count = _local_search(best_edges, seed=30000, time_budget=remaining)
        if final_count > best_count:
            best_count = final_count
            best_edges = final_edges[:]
    
    # Final validation
    if not _validate(best_edges):
        # Fallback: use greedy result
        best_edges = greedy_results[0][1]
        best_count = len(best_edges)
    
    return [[u, v] for u, v in best_edges]
