import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    m = len(edges)
    
    # Precompute edge indices for each vertex
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For n <= 15, brute force is faster and simpler.
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for e in edges:
                val = (mask >> e[0]) & 1
                mono = True
                for v in e[1:]:
                    if ((mask >> v) & 1) != val:
                        mono = False
                        break
                if mono:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    # For n > 15, use backtracking with unit propagation
    colors = [-1] * n
    edge_counts = [[0, 0] for _ in range(m)]
    
    # Vertex ordering: sort by degree descending for better pruning
    vertex_order = sorted(range(n), key=lambda v: len(vertex_edges[v]), reverse=True)
    
    def backtrack():
        # Check for conflicts and find forced assignments
        target_vertex = -1
        target_color = -1
        forced = False
        
        for i in range(m):
            c0, c1 = edge_counts[i]
            if c0 == 5 or c1 == 5:
                return False
            if c0 == 4 and c1 == 0:
                for v in edges[i]:
                    if colors[v] == -1:
                        target_vertex = v
                        target_color = 1
                        forced = True
                        break
            elif c1 == 4 and c0 == 0:
                for v in edges[i]:
                    if colors[v] == -1:
                        target_vertex = v
                        target_color = 0
                        forced = True
                        break
            if forced:
                break
        
        if not forced:
            # Find any unassigned vertex, preferring higher degree
            for v in vertex_order:
                if colors[v] == -1:
                    target_vertex = v
                    break
            if target_vertex == -1:
                return True
        
        # Apply assignment
        colors[target_vertex] = target_color
        affected_edges = vertex_edges[target_vertex]
        saved_counts = []
        for i in affected_edges:
            saved_counts.append((edge_counts[i][0], edge_counts[i][1]))
            if target_color == 0:
                edge_counts[i][0] += 1
            else:
                edge_counts[i][1] += 1
        
        if backtrack():
            return True
        
        # Undo
        colors[target_vertex] = -1
        for i, (sc0, sc1) in zip(affected_edges, saved_counts):
            edge_counts[i][0] = sc0
            edge_counts[i][1] = sc1
        
        if forced:
            return False
        
        # Try the other color
        other_color = 1 - target_color
        colors[target_vertex] = other_color
        saved_counts = []
        for i in affected_edges:
            saved_counts.append((edge_counts[i][0], edge_counts[i][1]))
            if other_color == 0:
                edge_counts[i][0] += 1
            else:
                edge_counts[i][1] += 1
        
        if backtrack():
            return True
            
        # Undo
        colors[target_vertex] = -1
        for i, (sc0, sc1) in zip(affected_edges, saved_counts):
            edge_counts[i][0] = sc0
            edge_counts[i][1] = sc1
            
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Start with K9^5 (126 edges)
    n_start = 9
    edges_start = [list(e) for e in combinations(range(n_start), 5)]
    best_edges = edges_start[:]
    best_n = n_start
    
    # Try to shrink using greedy edge removal
    def shrink(n, edges, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        current_edges = edges[:]
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing edges in random order
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            for i in indices:
                if time.time() >= time_limit:
                    break
                # Try removing edge i
                new_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, new_edges):
                    current_edges = new_edges
                    improved = True
                    break
        return current_edges
    
    # Run shrink multiple times with different random seeds
    for attempt in range(10):
        if time.time() > deadline:
            break
        shrunk = shrink(best_n, best_edges)
        if len(shrunk) < len(best_edges):
            best_edges = shrunk
            # Continue shrinking if we made progress
        elif time.time() > deadline - 30:
            break
    
    # Try random search on different n values
    # Focus on n=9-13 where good constructions exist
    for n_try in [9, 10, 11, 12, 13]:
        if time.time() > deadline - 10:
            break
        # Try different edge counts, starting from reasonable lower bounds
        # For 5-uniform, non-2-colorable requires at least ~n*log(n) edges
        for m_try in range(30, 80, 3):
            if time.time() > deadline - 5:
                break
            # Try a few random instances
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                # Generate random hypergraph
                all_possible = list(combinations(range(n_try), 5))
                random.shuffle(all_possible)
                candidate_edges = all_possible[:m_try]
                if not is_2_colorable(n_try, candidate_edges):
                    # Found a non-2-colorable one. Shrink it.
                    remaining_time = deadline - time.time()
                    if remaining_time > 2:
                        shrunk = shrink(n_try, candidate_edges, time_limit=time.time() + remaining_time - 2)
                    else:
                        shrunk = candidate_edges
                    if len(shrunk) < len(best_edges):
                        best_edges = shrunk[:]
                        best_n = n_try
                    break # Found one, try next m or n
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    # Verify
    print("Verifying...")
    assert not is_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
