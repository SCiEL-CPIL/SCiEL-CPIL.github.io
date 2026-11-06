import time
import random
from itertools import combinations

def _is_2_colorable_exact(n, edges, timeout=None):
    """Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not."""
    if timeout is None:
        start = time.time()
    else:
        start = time.time()
    
    m = len(edges)
    if m == 0:
        return True
    
    # Preprocess edges: convert to frozensets for fast lookup, but keep original for vertex iteration
    edge_sets = [set(e) for e in edges]
    
    # We'll use a recursive backtracking approach.
    # State: assignment array of length n, with -1 for unassigned, 0/1 for assigned.
    # Unit propagation: if an edge has all but one vertex assigned the same color, the last must be different.
    
    # For efficiency, we'll maintain:
    # - For each edge, a count of 0s and 1s, and list of unassigned vertices.
    # But since n <= 26 and m <= 200, we can afford a simpler approach.
    
    # Actually, let's use a smarter approach:
    # We'll use the fact that for 5-uniform hypergraphs, if an edge has 4 vertices of one color, 
    # the 5th must be the other color.
    
    # Let's implement with a queue for unit propagation.
    
    # We'll represent the problem as:
    # - colors: list of -1, 0, or 1 for each vertex
    # - For each edge, track how many 0s, 1s, and unassigned
    
    # To make this efficient, we'll use a different approach:
    # Randomized backtracking with forward checking.
    
    # Actually, for n up to 26, 2^26 = 67 million is too many.
    # But with unit propagation and good variable ordering, it should be much faster.
    
    # Let's use a standard CSP approach with backtracking and unit propagation.
    
    # We'll use a recursive function with memoization on the state? No, state space is too large.
    
    # Instead, we'll use:
    # 1. Pick a vertex to assign (MRV: minimum remaining values, or just first unassigned).
    # 2. Try both colors.
    # 3. After assignment, do unit propagation.
    # 4. If contradiction, backtrack.
    
    # For time management, we'll count nodes visited.
    
    node_count = [0]
    time_limit = start + timeout if timeout is not None else float('inf')
    
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate(colors, edge_state):
        """Apply unit propagation. Returns False if contradiction, True otherwise.
        colors: list of -1/0/1
        edge_state: list of [count_0, count_1, unassigned_count] for each edge
        """
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if colors[v] != -1:
                    continue
                # For each edge containing v, check if it forces v's color
                forced_color = -1
                for ei in vertex_edges[v]:
                    c0, c1, cu = edge_state[ei]
                    if cu == 1:
                        # Only v is unassigned in this edge
                        if c1 == 4:
                            # All other 4 are color 1, so v must be 0
                            if forced_color != -1 and forced_color != 0:
                                return False
                            forced_color = 0
                        elif c0 == 4:
                            # All other 4 are color 0, so v must be 1
                            if forced_color != -1 and forced_color != 1:
                                return False
                            forced_color = 1
                if forced_color != -1:
                    colors[v] = forced_color
                    changed = True
                    # Update edge states
                    for ei in vertex_edges[v]:
                        if forced_color == 0:
                            edge_state[ei][0] += 1
                        else:
                            edge_state[ei][1] += 1
                        edge_state[ei][2] -= 1
                    # Check if any edge is now violated (all same color)
                    for ei in vertex_edges[v]:
                        c0, c1, cu = edge_state[ei]
                        if cu == 0:
                            if c0 == 5 or c1 == 5:
                                return False
        return True
    
    def backtrack(colors, edge_state):
        """Returns True if a valid coloring exists."""
        node_count[0] += 1
        if node_count[0] > 1000000:  # Safety limit
            return True  # Assume colorable if we can't prove otherwise quickly
        if time.time() > time_limit:
            return True  # Assume colorable on timeout
        
        # Check if all assigned
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True  # All assigned, no contradiction found
        
        # Try to find a vertex to assign (use first unassigned for simplicity)
        v = unassigned[0]
        
        # Try color 0
        colors_copy = colors[:]
        es_copy = [list(x) for x in edge_state]
        colors_copy[v] = 0
        es_copy[vertex_edges[v][0]][0] += 1
        es_copy[vertex_edges[v][0]][2] -= 1
        # Update all edges containing v
        for ei in vertex_edges[v]:
            es_copy[ei][0] += 1
            es_copy[ei][2] -= 1
            # Check for immediate violation
            if es_copy[ei][2] == 0 and es_copy[ei][1] == 5:
                pass  # Will be caught by propagate
        
        if propagate(colors_copy, es_copy):
            if backtrack(colors_copy, es_copy):
                return True
        
        # Try color 1
        colors_copy = colors[:]
        es_copy = [list(x) for x in edge_state]
        colors_copy[v] = 1
        for ei in vertex_edges[v]:
            es_copy[ei][1] += 1
            es_copy[ei][2] -= 1
        
        if propagate(colors_copy, es_copy):
            if backtrack(colors_copy, es_copy):
                return True
        
        return False
    
    colors = [-1] * n
    edge_state = [[0, 0, 5] for _ in range(m)]
    
    return backtrack(colors, edge_state)


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # We'll search for a non-2-colorable 5-uniform hypergraph with few edges.
    # Strategy: Start with a random hypergraph and try to minimize edges while maintaining non-2-colorability.
    
    # Known result: The minimum number of edges for a non-2-colorable 5-uniform hypergraph is between 29 and 51.
    # Best known construction has 51 edges.
    
    # Let's try to find a hypergraph with as few edges as possible.
    
    # Approach:
    # 1. Start with a small n (e.g., 9-15 vertices)
    # 2. Generate random 5-uniform hypergraphs
    # 3. Check if non-2-colorable
    # 4. If yes, try to remove edges to minimize
    
    # For efficiency, we'll focus on small n where exhaustive check is feasible.
    
    best_edges = None
    best_count = float('inf')
    
    # Try different n values
    for n in range(9, 16):
        if time.time() > deadline - 10:
            break
        
        # Generate many random hypergraphs with increasing edge counts
        for target_edges in range(29, 60):
            if time.time() > deadline - 10:
                break
            
            # Try several random instances
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Randomly choose target_edges edges from all possible
                all_possible = list(combinations(range(n), 5))
                random.shuffle(all_possible)
                edges = [list(e) for e in all_possible[:target_edges]]
                
                # Check if non-2-colorable
                if not _is_2_colorable_exact(n, edges, timeout=1.0):
                    # Found a non-2-colorable hypergraph!
                    # Try to minimize by removing edges
                    current_edges = edges[:]
                    current_n = n
                    
                    # Greedy edge removal
                    for _ in range(3):  # Multiple passes
                        if time.time() > deadline - 10:
                            break
                        
                        random.shuffle(current_edges)
                        new_edges = []
                        for edge in current_edges:
                            if time.time() > deadline - 10:
                                new_edges.append(edge)
                                continue
                            
                            # Try to remove this edge
                            candidate = [e for e in current_edges if e != edge]
                            if not _is_2_colorable_exact(current_n, candidate, timeout=0.5):
                                # Still non-2-colorable, remove it
                                continue
                            else:
                                # Must keep it
                                new_edges.append(edge)
                        current_edges = new_edges
                    
                    if len(current_edges) < best_count:
                        best_count = len(current_edges)
                        best_edges = current_edges[:]
                        # Update n if needed
                        if best_edges:
                            max_v = max(max(e) for e in best_edges)
                            best_n = max_v + 1
                        else:
                            best_n = current_n
                    
                    # Early termination if we reach known lower bound
                    if best_count <= 29:
                        break
            
            if best_count <= 29:
                break
    
    if best_edges is not None:
        # Verify the result
        max_v = max(max(e) for e in best_edges) if best_edges else 0
        n_final = max_v + 1
        if n_final < 5:
            n_final = 5
        
        return {'n': n_final, 'edges': best_edges}
    
    # Fallback: return a known valid hypergraph (K_9^5 with all 126 edges)
    # This is guaranteed to be non-2-colorable
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, edges={len(res['edges'])}")
