import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Fast check for 2-colorability.
    Returns True if 2-colorable, False otherwise.
    """
    # Pre-compute edge indices for vertices?
    # Let's use a simple recursive approach with memoization or just fast backtracking.
    
    # To be very fast, we can use a SAT solver approach or just aggressive pruning.
    # Given n <= 26, we can represent coloring as a tuple or list.
    
    # Optimization: Order edges by size? All are size 5.
    # Heuristic: Pick the edge with the most constrained vertices.
    
    # Let's try a different approach: 
    # If the hypergraph is not 2-colorable, it contains an odd cycle in some sense? No, that's for graphs.
    # For hypergraphs, it's more complex.
    
    # We will use a backtracking algorithm with unit propagation.
    # To speed it up, we will stop as soon as we find a valid coloring (returns True).
    # If we exhaust all possibilities, it's not 2-colorable (returns False).
    
    # We need to be careful with the time limit.
    # If the search takes too long, we can't distinguish between "not 2-colorable" and "too slow".
    # However, for small m (29-50), the search space is manageable if we use good heuristics.
    
    # Heuristic: Choose the unassigned vertex that appears in the most "almost satisfied" edges.
    
    # Let's implement a fast version.
    
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    
    def propagate(coloring):
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                edge = edges[i]
                c0 = 0
                c1 = 0
                unknown = -1
                for v in edge:
                    if coloring[v] == 0:
                        c0 += 1
                    elif coloring[v] == 1:
                        c1 += 1
                    else:
                        unknown = v
                
                if c0 == 4 and unknown != -1:
                    coloring[unknown] = 1
                    changed = True
                elif c1 == 4 and unknown != -1:
                    coloring[unknown] = 0
                    changed = True
                elif c0 == 5 or c1 == 5:
                    return None
        return coloring

    def backtrack(coloring):
        coloring = propagate(coloring[:])
        if coloring is None:
            return None
        
        # Find unassigned vertex
        unassigned = [v for v in range(n) if coloring[v] == -1]
        if not unassigned:
            return coloring
        
        # Heuristic: Pick vertex in the edge with the most assigned vertices?
        # Or just the one with highest degree.
        # Let's pick the one with the most edges.
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        for c in [0, 1]:
            coloring[v] = c
            res = backtrack(coloring)
            if res is not None:
                return res
            coloring[v] = -1
        return None

    return backtrack([-1]*n) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a hypergraph with < 51 edges
    # Search space: n from 9 to 15, m from 29 to 50
    
    # To make it faster, we can try specific structures or just random.
    # Random search with fast checker.
    
    best = None
    
    # Fallback: A known 51-edge construction.
    # We can generate one by taking a random 5-uniform hypergraph on 11 or 12 vertices 
    # and checking it. But we need to be sure.
    # Actually, the prompt says "best known construction has 51 edges". 
    # I will assume that if I don't find < 51, I should return a valid 51-edge one.
    # But generating a guaranteed 51-edge one is tricky without hardcoding.
    # Let's try to find < 51 first.
    
    # Optimize search:
    # For each m in [29, 30, ..., 50]:
    #   For each n in [9, 10, ..., 15]:
    #     Try random hypergraphs.
    
    # To increase chance of success, we can use a specific random seed or multiple trials.
    
    for m in range(29, 51):
        if time.time() > deadline:
            break
        for n in range(9, 16):
            if time.time() > deadline:
                break
            # Number of trials per (n, m)
            trials = 10
            for t in range(trials):
                if time.time() > deadline:
                    break
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                
                # Check
                if not is_2_colorable_fast(n, edges):
                    # Found one!
                    return {'n': n, 'edges': edges}
    
    # If not found < 51, we need to return a valid hypergraph.
    # The seed program returned 51 edges (score 1.0).
    # We should try to find a 51-edge one quickly to ensure validity.
    # Or just return the best found so far? No, if we found nothing, we must return something valid.
    # Let's try to find a 51-edge one with a few more trials.
    
    for m in [51]:
        if time.time() > deadline:
            break
        for n in range(10, 16):
            if time.time() > deadline:
                break
            trials = 50
            for t in range(trials):
                if time.time() > deadline:
                    break
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                if not is_2_colorable_fast(n, edges):
                    return {'n': n, 'edges': edges}
    
    # Emergency fallback: Complete hypergraph on 9 vertices (126 edges).
    # This is valid but score will be low (1.0 * 51/126 ~ 0.4).
    # But it's better than crashing.
    # However, we should try harder to find a small one.
    
    # Let's try a deterministic construction if random fails.
    # E.g., all edges containing vertex 0? No, that's star, 2-colorable.
    
    # If we reach here, we failed to find a small one.
    # We must return a valid hypergraph.
    # Let's use the complete hypergraph on 9 vertices as a last resort.
    edges = [list(e) for e in combinations(range(9), 5)]
    return {'n': 9, 'edges': edges}
