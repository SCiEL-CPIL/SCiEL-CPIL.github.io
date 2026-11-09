import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _find_all_maximal_cliques(adj, n, max_size=6):
    """Find all maximal cliques up to max_size using Bron-Kerbosch with pivot."""
    cliques = []
    R = set()
    P = set(range(n))
    X = set()
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        if len(R) >= max_size:
            # Still try to expand if we haven't hit the limit
            if len(R) < max_size:
                pass
            else:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
        
        # Choose pivot with maximum degree in P
        if P:
            pivot = max(P, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            R.add(v)
            bron_kerbosch(R, P & adj[v], X & adj[v])
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    bron_kerbosch(R, P, X)
    return cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edge_set = set(_edge_key(u, v) for u, v in edges)
    
    # Find all maximal cliques up to size 5 (triangles, K4s, K5s)
    # For dense graphs, these are the main building blocks
    all_cliques = _find_all_maximal_cliques(adj, n, max_size=5)
    
    # Filter to only cliques that are fully present in the graph
    valid_cliques = []
    for clique in all_cliques:
        clique_list = sorted(clique)
        is_clique = True
        for i, u in enumerate(clique_list):
            for v in clique_list[i+1:]:
                if _edge_key(u, v) not in edge_set:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique and len(clique) >= 2:
            valid_cliques.append(clique)
    
    if not valid_cliques:
        return [[u, v] for u, v in edges]
    
    # Build edge-to-clique mapping
    clique_edge_sets = []
    for clique in valid_cliques:
        edge_set_for_clique = set()
        for u, v in combinations(sorted(clique), 2):
            edge_set_for_clique.add(_edge_key(u, v))
        clique_edge_sets.append(edge_set_for_clique)
    
    num_cliques = len(valid_cliques)
    
    # Build conflict graph: two cliques conflict if they share an edge
    # For Lagrangian relaxation, we'll use a different approach:
    # Greedily select cliques that maximize uncovered edges covered
    
    best_partition = None
    best_count = float('inf')
    
    # Multiple random restarts with different orderings
    for run_seed in range(30):
        if not _deadline_ok():
            break
        
        rng = random.Random(seed + run_seed)
        
        # Start with all edges uncovered
        uncovered_edges = set(edge_set)
        selected_cliques = []
        
        # Greedy: repeatedly pick the clique that covers the most uncovered edges
        while uncovered_edges:
            if not _deadline_ok():
                # Fallback: add remaining edges as K2
                for e in uncovered_edges:
                    selected_cliques.append([e[0], e[1]])
                break
            
            # Find clique with maximum uncovered edges
            best_clique_idx = -1
            best_coverage = 0
            tie_list = []
            
            for idx, clique in enumerate(valid_cliques):
                coverage = 0
                for e in clique_edge_sets[idx]:
                    if e in uncovered_edges:
                        coverage += 1
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique_idx = idx
                    tie_list = [idx]
                elif coverage == best_coverage and coverage > 0:
                    tie_list.append(idx)
            
            if not tie_list or best_coverage == 0:
                # No clique covers any remaining edge; fallback to K2
                for e in uncovered_edges:
                    selected_cliques.append([e[0], e[1]])
                break
            
            # Random tie-breaking
            chosen_idx = rng.choice(tie_list)
            clique = valid_cliques[chosen_idx]
            selected_cliques.append(sorted(clique))
            
            # Remove covered edges
            for e in clique_edge_sets[chosen_idx]:
                if e in uncovered_edges:
                    uncovered_edges.remove(e)
        
        count = len(selected_cliques)
        if count < best_count:
            best_count = count
            best_partition = selected_cliques
        
        # Also try a variant: prioritize larger cliques first with some randomness
        if not _deadline_ok():
            break
            
        rng2 = random.Random(seed + run_seed + 1000)
        uncovered_edges2 = set(edge_set)
        selected_cliques2 = []
        
        while uncovered_edges2:
            if not _deadline_ok():
                for e in uncovered_edges2:
                    selected_cliques2.append([e[0], e[1]])
                break
            
            # Score = (coverage * 10 + size) with noise
            best_score = -1
            best_idx = -1
            tie_list = []
            
            for idx, clique in enumerate(valid_cliques):
                coverage = 0
                for e in clique_edge_sets[idx]:
                    if e in uncovered_edges2:
                        coverage += 1
                if coverage == 0:
                    continue
                size = len(clique)
                score = coverage * 10 + size + rng2.random() * 2
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    tie_list = [idx]
                elif abs(score - best_score) < 0.01:
                    tie_list.append(idx)
            
            if not tie_list:
                for e in uncovered_edges2:
                    selected_cliques2.append([e[0], e[1]])
                break
            
            chosen_idx = rng2.choice(tie_list)
            clique = valid_cliques[chosen_idx]
            selected_cliques2.append(sorted(clique))
            
            for e in clique_edge_sets[chosen_idx]:
                if e in uncovered_edges2:
                    uncovered_edges2.remove(e)
        
        count2 = len(selected_cliques2)
        if count2 < best_count:
            best_count = count2
            best_partition = selected_cliques2
    
    if best_partition is None:
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
