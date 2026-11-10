# EVOLVE-BLOCK-START
import time
import random

def _generate_graph(n, p, seed):
    """Generate the same graph as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_cliques(n, adj, time_limit=5.0):
    """Find maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    start = time.time()
    
    def bronk(R, P, X):
        if time.time() - start > time_limit:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        candidates = P | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(adj[v] & P))
            candidates = candidates - adj[pivot]
        else:
            candidates = set()
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bronk(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}
    
    bronk(frozenset(), set(range(n)), frozenset())
    return cliques

def _solve_instance(n, p, seed, time_budget):
    """Solve one instance: partition edges into minimum number of cliques."""
    edges = _generate_graph(n, p, seed)
    adj = _build_adjacency(n, edges)
    edge_set = set(edges)
    
    # Find maximal cliques
    max_cliques = _find_maximal_cliques(n, adj, time_limit=min(3.0, time_budget * 0.2))
    
    # Build candidate list: maximal cliques + all edges + some 3-cliques
    candidates = []
    
    # Maximal cliques
    for c in max_cliques:
        if len(c) >= 2:
            candidates.append(frozenset(c))
    
    # All edges as 2-cliques
    for e in edges:
        candidates.append(frozenset(e))
    
    # 3-cliques
    for u in range(n):
        for v in sorted(adj[u]):
            if v > u:
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        candidates.append(frozenset((u, v, w)))
    
    # Precompute edge sets for each candidate
    def clique_edges(clique):
        s = list(clique)
        es = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                a, b = s[i], s[j]
                es.add((min(a,b), max(a,b)))
        return frozenset(es)
    
    candidates_info = []
    for c in candidates:
        ce = clique_edges(c)
        if ce:
            candidates_info.append((c, ce))
    
    # Greedy set cover with multiple strategies
    def greedy_cover(candidates_info, edge_set, rng=None):
        uncovered = set(edge_set)
        selected = []
        remaining = list(range(len(candidates_info)))
        
        # If rng, shuffle
        if rng:
            rng.shuffle(remaining)
        
        # Sort by clique size descending (larger cliques first)
        remaining.sort(key=lambda i: -len(candidates_info[i][0]))
        
        while uncovered:
            best_idx = -1
            best_count = 0
            for idx in remaining:
                c, ce = candidates_info[idx]
                cnt = len(ce & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx == -1 or best_count == 0:
                # Should not happen since we have 2-cliques
                break
            c, ce = candidates_info[best_idx]
            selected.append(list(c))
            uncovered -= ce
            remaining.remove(best_idx)
        
        # Add any remaining uncovered edges
        if uncovered:
            for e in uncovered:
                selected.append(list(e))
        
        return selected
    
    # Try initial greedy
    best_selected = greedy_cover(candidates_info, edge_set)
    best_count = len(best_selected)
    
    # Try random restarts
    rng = random.Random(123)
    num_restarts = min(50, int(time_budget * 2))
    
    for _ in range(num_restarts):
        # Randomize: shuffle candidates before sorting by size
        # Or use a random subset
        shuffled_info = candidates_info[:]
        rng.shuffle(shuffled_info)
        # Re-sort by size
        shuffled_info.sort(key=lambda x: -len(x[0]))
        
        sel = greedy_cover(shuffled_info, edge_set)
        if len(sel) < best_count:
            best_count = len(sel)
            best_selected = sel
    
    # Local improvement: try to replace subsets of selected cliques
    # with a single larger clique
    # For each selected clique, check if it can be extended
    # This is complex, skip for now
    
    return best_selected

def construct():
    start_time = time.time()
    total_budget = 110.0  # Safety margin before 120s limit
    
    instances = [(32, 0.6, 601), (64, 0.55, 602)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 5.0:
            # Not enough time, use trivial solution
            edges = _generate_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        # Allocate time: first instance gets ~40%, second gets rest
        if i == 0:
            budget = min(remaining * 0.35, 30.0)
        else:
            budget = min(remaining - 2.0, 105.0)
        
        result = _solve_instance(n, p, seed, budget)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
