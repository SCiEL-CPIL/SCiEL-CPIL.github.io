"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the book size for each edge in the graph."""
    adj_list = defaultdict(set)
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    max_book = 0
    for u, v in edges:
        common = adj_list[u] & adj_list[v]
        book_size = len(common)
        if book_size > max_book:
            max_book = book_size
    
    return max_book

def _simulate_annealing(n=40, target_edges=401, time_limit=140.0, seed=42):
    """Simulated annealing to minimize the maximum book size."""
    start_time = time.time()
    random.seed(seed)
    
    # Initialize with a random graph with exactly target_edges edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    current_edges = all_possible[:target_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute book sizes
    book_sizes = {}
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        book_sizes[key] = len(adj[u] & adj[v])
    
    current_max = max(book_sizes.values()) if book_sizes else 0
    
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    # SA parameters
    T0 = 5.0
    Tend = 0.01
    alpha = 0.999
    
    T = T0
    step = 0
    while T > Tend and time.time() - start_time < time_limit:
        step += 1
        
        # Pick a random edge to remove
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        # Pick a random non-edge to add
        cu, cv = random.choice(all_possible)
        ckey = (min(cu, cv), max(cu, cv))
        if ckey in edge_set:
            continue
        
        # Compute book size of new edge
        new_book = len(adj[cu] & adj[cv])
        
        # Remove old edge, add new edge
        old_book = book_sizes[rkey]
        
        # Update structures
        del book_sizes[rkey]
        book_sizes[ckey] = new_book
        edge_set.remove(rkey)
        edge_set.add(ckey)
        adj[ru].remove(rv)
        adj[rv].remove(ru)
        adj[cu].add(cv)
        adj[cv].add(cu)
        current_edges[remove_idx] = [cu, cv]
        
        # Recompute max
        new_max = max(book_sizes.values()) if book_sizes else 0
        
        # Delta in energy (we want to minimize)
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
        else:
            # Revert
            book_sizes[rkey] = old_book
            del book_sizes[ckey]
            edge_set.remove(ckey)
            edge_set.add(rkey)
            adj[cu].remove(cv)
            adj[cv].remove(cu)
            adj[ru].add(rv)
            adj[rv].add(ru)
            current_edges[remove_idx] = [ru, rv]
        
        T *= alpha
    
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 145.0  # Safety margin
    
    # Try multiple SA runs with different seeds
    best_edges = None
    best_score = float('inf')
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 5.0:
            break
        time_limit = min(25.0, deadline - time.time() - 1.0)
        if time_limit < 2.0:
            break
        edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    if best_edges is None:
        # Fallback: bipartite + 1 edge
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        best_score = _compute_book_sizes(best_edges, n)
    
    return best_edges
# EVOLVE-BLOCK-END
