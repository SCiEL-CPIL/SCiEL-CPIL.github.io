"""Construct dense 4-cycle-rainbow edge-coloured graph on 20 vertices."""
import time
import random
import sys
from itertools import combinations

N = 20
N_COLORS = 20
DEADLINE = 150.0
SAFETY = 15.0  # safety margin in seconds

def _validate(edges):
    """Check that every 4-cycle has four distinct colours."""
    if not edges:
        return True
    adj = {}
    for u, v, c in edges:
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c

    verts = list(range(N))
    for a, b, c, d in combinations(verts, 4):
        cycles = [
            [(a,b), (b,c), (c,d), (d,a)],
            [(a,b), (b,d), (d,c), (c,a)],
            [(a,c), (c,b), (b,d), (d,a)],
        ]
        for cyc in cycles:
            cols = []
            valid_cycle = True
            for (u, v) in cyc:
                if v not in adj.get(u, {}):
                    valid_cycle = False
                    break
                cols.append(adj[u][v])
            if valid_cycle:
                if len(set(cols)) != 4:
                    return False
    return True

def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start

    def deadline_ok():
        return elapsed() < DEADLINE - SAFETY

    # Precompute all possible edges
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    # Fast greedy builder with forbidden-color computation
    def greedy_build(seed, initial_edges=None, order_type='random'):
        rng = random.Random(seed)
        
        # adj[u][v] = color, -1 if no edge
        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        edges = []
        
        if initial_edges:
            for u, v, c in initial_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing_set = set((min(u,v), max(u,v)) for u,v,_ in edges)
            remaining = [e for e in all_pairs if e not in existing_set]
        else:
            remaining = all_pairs[:]
            
        if order_type == 'random':
            rng.shuffle(remaining)
        elif order_type == 'reverse':
            remaining.reverse()
            
        for (u, v) in remaining:
            if v in neighbors[u]:
                continue
            
            # Find forbidden colors: any 4-cycle u-a-b-v where (a,b) exists
            # The colors on (u,a), (a,b), (b,v) are all forbidden for (u,v)
            # Also, if those three colors are not all distinct, the edge is impossible
            impossible = False
            forbidden = set()
            
            for a in list(neighbors[u]):
                if a == v:
                    continue
                ca = adj[u][a]
                for b in list(neighbors[v]):
                    if b == u or b == a:
                        continue
                    if adj[a][b] == -1:
                        continue
                    cb = adj[a][b]
                    cv = adj[b][v]
                    # If any two of ca, cb, cv are equal, adding (u,v) creates a non-rainbow cycle
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
                if impossible:
                    break
            
            if not impossible:
                colors = list(range(N_COLORS))
                rng.shuffle(colors)
                for c in colors:
                    if c not in forbidden:
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        break
        
        return edges

    best_edges = None
    best_count = 0
    
    # Phase 1: Multiple restarts with different orderings
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 150
    
    for ot in order_types:
        for i in range(num_restarts):
            if not deadline_ok():
                break
            edges = greedy_build(i * 7 + 13, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    if best_edges is None:
        # Fallback: simple star
        best_edges = [[0, v, v % N_COLORS] for v in range(1, N)]
        best_count = len(best_edges)
    
    # Phase 2: Local search refinement
    # Remove k random edges and re-greedy-fill to escape local optima
    current_best = best_edges[:]
    max_refine = 3000
    refine_iter = 0
    
    while deadline_ok() and refine_iter < max_refine:
        refine_iter += 1
        
        # Choose k edges to remove (1 to 6)
        k = random.randint(1, 6)
        if len(current_best) <= k:
            continue
        
        # Randomly select edges to remove
        edges_to_remove = random.sample(current_best, k)
        remove_set = set((min(u,v), max(u,v)) for u,v,_ in edges_to_remove)
        
        # Keep the rest
        base_edges = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
        
        # Re-run greedy from this base
        seed = refine_iter * 1000 + random.randint(0, 1000)
        new_edges = greedy_build(seed, base_edges, 'random')
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
            current_best = new_edges[:]
    
    # Final validation
    if not _validate(best_edges):
        # Fallback to a known valid small graph
        return [[0, v, v % N_COLORS] for v in range(1, N)]
    
    return best_edges

if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Valid: {_validate(result)}")
