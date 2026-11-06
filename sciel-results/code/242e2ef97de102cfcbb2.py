import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Map vertices to edges
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
    
    # State: color of each vertex (0 or 1), or -1 if uncolored
    color = [-1] * n
    # For each edge, count how many vertices are colored 0 and 1
    edge_counts = [[0, 0] for _ in range(len(edges))]
    
    def propagate():
        """
        Propagate constraints. Returns False if a contradiction is found.
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0, c1 = edge_counts[i]
                uncolored = [v for v in e if color[v] == -1]
                if c0 == 5:
                    # All vertices must be 1
                    for v in uncolored:
                        if color[v] == 0:
                            return False
                        color[v] = 1
                        edge_counts[i][1] += 1
                        changed = True
                elif c1 == 5:
                    # All vertices must be 0
                    for v in uncolored:
                        if color[v] == 1:
                            return False
                        color[v] = 0
                        edge_counts[i][0] += 1
                        changed = True
                elif c0 == 4 and len(uncolored) == 1:
                    # The last vertex must be 1
                    v = uncolored[0]
                    if color[v] == 0:
                        return False
                    color[v] = 1
                    edge_counts[i][1] += 1
                    changed = True
                elif c1 == 4 and len(uncolored) == 1:
                    # The last vertex must be 0
                    v = uncolored[0]
                    if color[v] == 1:
                        return False
                    color[v] = 0
                    edge_counts[i][0] += 1
                    changed = True
        return True
    
    def backtrack(idx):
        """
        Backtrack on edges.
        """
        if idx == len(edges):
            return True
        
        e = edges[idx]
        c0, c1 = edge_counts[idx]
        uncolored = [v for v in e if color[v] == -1]
        
        if not uncolored:
            # All vertices in this edge are colored
            if c0 == 5 or c1 == 5:
                return False
            else:
                return backtrack(idx + 1)
        
        # Try to assign colors to uncolored vertices in this edge
        # We can try to assign 0 or 1 to each uncolored vertex
        # But to avoid symmetry, we can fix the first uncolored vertex to 0
        first_v = uncolored[0]
        
        # Try 0
        color[first_v] = 0
        edge_counts[idx][0] += 1
        if propagate() and backtrack(idx + 1):
            return True
        color[first_v] = -1
        edge_counts[idx][0] -= 1
        
        # Try 1
        color[first_v] = 1
        edge_counts[idx][1] += 1
        if propagate() and backtrack(idx + 1):
            return True
        color[first_v] = -1
        edge_counts[idx][1] -= 1
        
        return False
    
    if not propagate():
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10 seconds margin
    
    best_edges = None
    best_n = None
    
    # Try to find a small non-2-colorable hypergraph
    # Start with a known dense hypergraph and try to remove edges
    # Or use randomized search
    
    # Let's try a randomized search
    # We'll try different n and m
    for n in range(9, 27):
        if time.time() > deadline:
            break
        for m in range(5, 200):
            if time.time() > deadline:
                break
            # Try a few random hypergraphs
            for _ in range(10):
                if time.time() > deadline:
                    break
                edges = []
                while len(edges) < m:
                    e = sorted(random.sample(range(n), 5))
                    if e not in edges:
                        edges.append(e)
                
                if not is_2_colorable(n, edges):
                    # Found a non-2-colorable hypergraph
                    # Try to remove edges to minimize m
                    while m > 5:
                        if time.time() > deadline:
                            break
                        # Try to remove a random edge
                        idx = random.randint(0, len(edges) - 1)
                        edges_removed = edges.pop(idx)
                        if is_2_colorable(n, edges):
                            edges.append(edges_removed)
                        else:
                            m -= 1
                    if best_edges is None or len(edges) < len(best_edges):
                        best_edges = edges[:]
                        best_n = n
    
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to the complete hypergraph on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
