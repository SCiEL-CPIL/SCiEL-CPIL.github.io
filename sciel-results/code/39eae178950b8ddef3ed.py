"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a dense edge-coloured graph on 20 vertices with 20 colours
    such that every 4-cycle is rainbow (all 4 edges have distinct colours).
    Maximize the number of edges.
    
    Strategy: Start with a dense graph (e.g., complement of a sparse graph or 
    near-complete), assign colours, then iteratively fix violations by either
    recoloring edges or removing edges. Use local search with simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n = 20
    num_colors = 20
    
    # Helper: check if a 4-cycle is rainbow given edge colours
    # A 4-cycle is u-v-w-x-u (or any 4 distinct vertices forming a cycle)
    # We need to enumerate all 4-cycles in the graph
    
    # Represent graph as adjacency with colours
    # adj[u][v] = colour or None if no edge
    
    def make_adj(edges):
        """edges is list of [u, v, c]"""
        adj = [[None]*n for _ in range(n)]
        for u, v, c in edges:
            if u < v:
                adj[u][v] = c
                adj[v][u] = c
            else:
                adj[v][u] = c
                adj[u][v] = c
        return adj
    
    def get_edges_from_adj(adj):
        """Get list of edges from adjacency matrix"""
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] is not None:
                    edges.append([u, v, adj[u][v]])
        return edges
    
    def find_all_4cycles(adj):
        """Find all 4-cycles in the graph. A 4-cycle is a set of 4 vertices {a,b,c,d}
        such that there are edges forming a cycle. We represent each 4-cycle by 
        its 4 edges in cyclic order."""
        cycles = []
        # For each set of 4 vertices, check if they form a 4-cycle
        # A 4-cycle on vertices a,b,c,d means edges (a,b), (b,c), (c,d), (d,a) exist
        # or some permutation forming a cycle
        
        # There are 3 ways to arrange 4 vertices in a cycle (up to rotation/reflection):
        # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
        
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b] is None:
                    continue
                for c in range(b+1, n):
                    if adj[b][c] is None:
                        continue
                    for d in range(c+1, n):
                        # Check possible 4-cycles
                        # Cycle: a-b-c-d-a
                        if adj[c][d] is not None and adj[d][a] is not None:
                            cycles.append([(a,b), (b,c), (c,d), (d,a)])
                        # Cycle: a-b-d-c-a  
                        if adj[b][d] is not None and adj[d][c] is not None and adj[c][a] is not None:
                            # Need to check this isn't the same cycle
                            # a-b-d-c-a uses edges (a,b), (b,d), (d,c), (c,a)
                            cycles.append([(a,b), (b,d), (d,c), (c,a)])
                        # Cycle: a-c-b-d-a
                        if adj[a][c] is not None and adj[c][b] is not None and adj[b][d] is not None and adj[d][a] is not None:
                            # a-c-b-d-a uses edges (a,c), (c,b), (b,d), (d,a)
                            # But we already have a-b edge... this would be a different structure
                            # Actually for 4 distinct vertices, the three distinct 4-cycles are:
                            # 1. a-b-c-d-a
                            # 2. a-b-d-c-a  
                            # 3. a-c-b-d-a
                            # But wait, some of these might use the same edges
                            cycles.append([(a,c), (c,b), (b,d), (d,a)])
        return cycles
    
    def is_rainbow(cycle, adj):
        """Check if a 4-cycle has all distinct colours"""
        colours = set()
        for u, v in cycle:
            c = adj[u][v]
            if c is None:
                return False  # Not actually a cycle in current graph
            if c in colours:
                return False
            colours.add(c)
        return True
    
    def count_violations(adj):
        """Count the number of non-rainbow 4-cycles"""
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b] is None:
                    continue
                for c in range(b+1, n):
                    if adj[b][c] is None:
                        continue
                    for d in range(c+1, n):
                        # Check the three possible 4-cycles
                        # 1. a-b-c-d-a
                        if adj[c][d] is not None and adj[d][a] is not None:
                            colours = {adj[a][b], adj[b][c], adj[c][d], adj[d][a]}
                            if len(colours) < 4:
                                violations += 1
                        # 2. a-b-d-c-a
                        if adj[b][d] is not None and adj[d][c] is not None and adj[c][a] is not None:
                            colours = {adj[a][b], adj[b][d], adj[d][c], adj[c][a]}
                            if len(colours) < 4:
                                violations += 1
                        # 3. a-c-b-d-a
                        if adj[a][c] is not None and adj[c][b] is not None and adj[b][d] is not None and adj[d][a] is not None:
                            colours = {adj[a][c], adj[c][b], adj[b][d], adj[d][a]}
                            if len(colours) < 4:
                                violations += 1
        return violations
    
    def num_edges(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] is not None:
                    count += 1
        return count
    
    # Strategy: Start with a moderately dense graph and use local search
    # to fix violations while maximizing edges
    
    # Initial construction: Use a structured approach
    # Start with all edges present, assign colours based on a pattern, then remove
    # edges that can't be fixed
    
    # Let's try: complete graph K20 with random initial colours, then iteratively
    # fix violations by recoloring or removing edges
    
    random.seed(42)
    
    # Start with K20
    adj = [[None]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            adj[u][v] = random.randint(0, num_colors-1)
            adj[v][u] = adj[u][v]
    
    # Simulated annealing: try to reduce violations while keeping edges
    # Move types:
    # 1. Recolor an edge
    # 2. Remove an edge
    # 3. Add an edge (with some colour)
    
    def objective(adj):
        """Objective: maximize edges - penalty * violations"""
        e = num_edges(adj)
        v = count_violations(adj)
        # We want to minimize violations first, then maximize edges
        # Use a large penalty for violations
        return -v * 1000 + e
    
    best_adj = [row[:] for row in adj]
    best_score = objective(best_adj)
    
    current_adj = [row[:] for row in adj]
    current_score = objective(current_adj)
    
    # Simulated annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    step = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > min_T:
        if no_improve > max_no_improve:
            # Reset temperature
            T = 5.0
            no_improve = 0
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Recolor an existing edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or current_adj[u][v] is None:
                continue
            old_color = current_adj[u][v]
            new_color = random.randint(0, num_colors-1)
            if new_color == old_color:
                continue
            current_adj[u][v] = new_color
            current_adj[v][u] = new_color
            
            new_score = objective(current_adj)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < (2.718281828 ** (delta / T)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_adj[u][v] = old_color
                current_adj[v][u] = old_color
        
        elif move_type < 0.85:
            # Remove an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or current_adj[u][v] is None:
                continue
            old_color = current_adj[u][v]
            current_adj[u][v] = None
            current_adj[v][u] = None
            
            new_score = objective(current_adj)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < (2.718281828 ** (delta / T)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_adj[u][v] = old_color
                current_adj[v][u] = old_color
        
        else:
            # Add an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or current_adj[u][v] is not None:
                continue
            new_color = random.randint(0, num_colors-1)
            current_adj[u][v] = new_color
            current_adj[v][u] = new_color
            
            new_score = objective(current_adj)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < (2.718281828 ** (delta / T)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in current_adj]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_adj[u][v] = None
                current_adj[v][u] = None
        
        T *= cooling
        step += 1
    
    # Final: try to add back edges that don't create violations
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] is None:
                # Try adding this edge with each colour
                for c in range(num_colors):
                    test_adj = [row[:] for row in best_adj]
                    test_adj[u][v] = c
                    test_adj[v][u] = c
                    if count_violations(test_adj) == count_violations(best_adj):
                        # No new violations
                        best_adj[u][v] = c
                        best_adj[v][u] = c
                        break
    
    # Verify final result
    final_violations = count_violations(best_adj)
    if final_violations > 0:
        # Need to fix remaining violations by removing edges
        # Find violating cycles and remove one edge from each
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if best_adj[a][b] is None:
                        continue
                    for c in range(b+1, n):
                        if best_adj[b][c] is None:
                            continue
                        for d in range(c+1, n):
                            # Check cycle a-b-c-d-a
                            if best_adj[c][d] is not None and best_adj[d][a] is not None:
                                colours = {best_adj[a][b], best_adj[b][c], best_adj[c][d], best_adj[d][a]}
                                if len(colours) < 4:
                                    # Remove the edge (a,b)
                                    best_adj[a][b] = None
                                    best_adj[b][a] = None
                                    changed = True
                                    break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        # If still violations, remove more edges aggressively
        while count_violations(best_adj) > 0 and time.time() < deadline:
            # Find a violating cycle and remove an edge
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    if best_adj[a][b] is None:
                        continue
                    for c in range(b+1, n):
                        if best_adj[b][c] is None:
                            continue
                        for d in range(c+1, n):
                            if best_adj[c][d] is not None and best_adj[d][a] is not None:
                                colours = {best_adj[a][b], best_adj[b][c], best_adj[c][d], best_adj[d][a]}
                                if len(colours) < 4:
                                    best_adj[a][b] = None
                                    best_adj[b][a] = None
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
    
    # Return the edges
    edges = get_edges_from_adj(best_adj)
    return edges

# EVOLVE-BLOCK-END
