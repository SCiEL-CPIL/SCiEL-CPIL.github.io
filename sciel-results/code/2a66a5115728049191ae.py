"""Clique partitioning of dense random graphs via multi-start greedy and local search."""
import time
from random import Random


def _random_graph(n, p, seed):
    """Generate random graph using the same RNG as the verifier."""
    rng = Random(seed)
    adj = [0] * n  # bitmask adjacency
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj


def _expand_clique(rem, start, n, rng=None, randomize=False):
    """
    Expand a clique starting from `start` in the graph defined by `rem`.
    """
    clique = [start]
    cand = rem[start]
    
    while cand:
        # Find best vertex in cand
        # Score: number of neighbors in cand
        best_u = -1
        best_score = -1
        
        # If randomizing, pick from top 3
        top_candidates = []
        
        temp = cand
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            score = (rem[u] & cand).bit_count()
            
            if randomize and rng:
                # Keep top 3 for random selection
                if len(top_candidates) < 3:
                    top_candidates.append((score, u))
                    if len(top_candidates) == 3:
                        top_candidates.sort(reverse=True)
                else:
                    if score > top_candidates[2][0]:
                        top_candidates[2] = (score, u)
                        top_candidates.sort(reverse=True)
            else:
                if score > best_score:
                    best_score = score
                    best_u = u
                    
            temp ^= low
            
        if randomize and rng:
            if top_candidates:
                chosen = rng.choice([u for _, u in top_candidates])
            else:
                break
        else:
            if best_u == -1:
                break
            chosen = best_u
            
        clique.append(chosen)
        cand &= rem[chosen]
        
    return clique if len(clique) >= 2 else None


def _greedy_partition(adj, n, rng_seed):
    """Find clique partition using greedy heuristic."""
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    rng = Random(rng_seed)
    
    if remaining_edges == 0:
        return cliques
        
    while remaining_edges > 0:
        # Find vertex with max degree
        max_deg = -1
        v = -1
        for i in range(n):
            d = rem[i].bit_count()
            if d > max_deg:
                max_deg = d
                v = i
        
        if v == -1 or max_deg == 0:
            break
        
        if max_deg == 1:
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Generate candidate cliques
        candidates = []
        
        # Deterministic max
        c1 = _expand_clique(rem, v, n, rng=None, randomize=False)
        if c1: candidates.append(c1)
        
        # Randomized variants
        for _ in range(3):
            c = _expand_clique(rem, v, n, rng=rng, randomize=True)
            if c: candidates.append(c)
            
        if not candidates:
            # Fallback
            low = rem[v] & (-rem[v])
            u = low.bit_length() - 1
            cliques.append([v, u])
            rem[v] &= ~(1 << u)
            rem[u] &= ~(1 << v)
            remaining_edges -= 1
            continue
        
        # Select best (largest)
        best_clique = max(candidates, key=len)
        m = len(best_clique)
        
        cliques.append(best_clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = best_clique[i], best_clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
        
    return cliques

def _local_search_merge(adj, n, cliques):
    """Try to merge cliques to reduce count."""
    # Build complement adjacency for quick clique check
    # comp[u] has bits set for v if (u,v) is NOT in adj
    comp = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                comp[u] |= (1 << v)
                comp[v] |= (1 << u)
                
    improved = True
    while improved:
        improved = False
        # Try merging all pairs
        # To avoid O(k^2) overhead if k is large, we can just do it. k is small.
        k = len(cliques)
        for i in range(k):
            if i >= len(cliques): break
            for j in range(i + 1, len(cliques)):
                V_i = set(cliques[i])
                V_j = set(cliques[j])
                
                # Check edge disjointness
                # Edges in i: all pairs in V_i
                # Edges in j: all pairs in V_j
                # Intersection is non-empty if there is an edge (u,v) in both.
                # This happens if |V_i \cap V_j| >= 2.
                # If |V_i \cap V_j| <= 1, edge sets are disjoint.
                if len(V_i & V_j) > 1:
                    continue
                
                V_new = V_i | V_j
                
                # Check if V_new is a clique in G
                # i.e., no edges in comp within V_new
                is_clique = True
                for u in V_new:
                    # Check neighbors of u in V_new
                    # If any neighbor v in V_new is in comp[u], then not a clique
                    neighbors_in_vnew = [v for v in V_new if v != u]
                    for v in neighbors_in_vnew:
                        if comp[u] & (1 << v):
                            is_clique = False
                            break
                    if not is_clique:
                        break
                
                if is_clique:
                    # Merge
                    cliques[i] = list(V_new)
                    cliques.pop(j)
                    improved = True
                    break
            if improved:
                break
    return cliques

def _solve_instance(adj, n, deadline):
    """Find best clique partition for a single graph."""
    best_cliques = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    start_time = time.time()
    max_runs = 50 # Adjust based on time
    run_id = 0
    
    while time.time() < deadline - 5: # Keep 5s for local search and finalization
        if run_id >= max_runs:
            break
        cliques = _greedy_partition(adj, n, run_id)
        count = len(cliques)
        
        if count < best_count:
            best_count = count
            best_cliques = cliques
            
        run_id += 1
        
    if best_cliques is None:
        # Fallback: single run
        best_cliques = _greedy_partition(adj, n, 0)
        best_count = len(best_cliques)
        
    # Local search: try to merge
    if time.time() < deadline - 1:
        best_cliques = _local_search_merge(adj, n, best_cliques)
        
    return best_cliques

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    
    return results
