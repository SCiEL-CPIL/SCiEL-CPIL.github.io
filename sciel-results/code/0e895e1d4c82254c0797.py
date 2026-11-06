import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
        
    # We will use a recursive solver that copies the assignment state.
    # Since n <= 26, copying a list of 26 ints is cheap.
    
    assignment = [-1] * n
    
    def propagate(assign):
        # Returns False if contradiction, True otherwise.
        # Modifies 'assign' in place.
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    cv = assign[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        unassigned = v
                        
                if c0 == 5 or c1 == 5:
                    return False
                    
                if c0 == 4 and unassigned != -1:
                    assign[unassigned] = 1
                    changed = True
                elif c1 == 4 and unassigned != -1:
                    assign[unassigned] = 0
                    changed = True
        return True

    def solve(assign):
        # Find next unassigned vertex
        v = -1
        # Heuristic: Pick vertex in an edge with most assigned vertices?
        # Simple: first unassigned.
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
                
        if v == -1:
            return True
            
        # Try 0
        new_assign_0 = assign[:]
        new_assign_0[v] = 0
        if propagate(new_assign_0):
            if solve(new_assign_0):
                return True
                
        # Try 1
        new_assign_1 = assign[:]
        new_assign_1[v] = 1
        if propagate(new_assign_1):
            if solve(new_assign_1):
                return True
                
        return False
        
    if not propagate(assignment):
        return False
        
    return solve(assignment)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Seed solution: Complete 5-uniform hypergraph on n=9 vertices.
    # This is non-2-colorable. Edges = C(9,5) = 126.
    seed_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = seed_edges
    best_n = 9
    best_count = len(best_edges)
    
    # We want to minimize edges.
    # Search n from 9 to 26.
    # For each n, try to find a non-2-colorable hypergraph with few edges.
    
    # Strategy:
    # 1. Start with a random set of edges.
    # 2. If it's 2-colorable, add edges until it's not.
    # 3. Greedily remove edges while maintaining non-2-colorability.
    
    for n in range(9, 27):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on remaining time
        # Let's do a few restarts per n
        restarts = 5
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a subset
            # Heuristic: Random 5-uniform hypergraphs become non-2-colorable around certain density.
            # For n=9, C(9,5)=126.
            # For larger n, we can start with fewer edges and add.
            
            # Let's start with k edges
            k = min(len(all_possible), 50 + random.randint(0, 50))
            edges = [list(e) for e in all_possible[:k]]
            
            # Ensure it's non-2-colorable
            # If it is, add random edges until it's not or we run out of time
            while is_2_colorable_fast(n, edges):
                if time.time() > deadline:
                    break
                if len(edges) < len(all_possible):
                    # Add a random edge not in edges
                    # Using a set for fast lookup
                    edge_set = set(tuple(e) for e in edges)
                    while True:
                        new_edge = tuple(random.choice(all_possible))
                        if new_edge not in edge_set:
                            edges.append(list(new_edge))
                            edge_set.add(new_edge)
                            break
                else:
                    # Complete graph, must be non-2-colorable for n>=9?
                    # Yes, as argued before.
                    break
            
            if time.time() > deadline:
                break
                
            # Verify it is non-2-colorable
            if is_2_colorable_fast(n, edges):
                continue
                
            # Greedy deletion
            # We want to remove edges one by one if possible.
            # Use a set for current edges
            current_edges = set(tuple(e) for e in edges)
            
            # Create a list of edges to try removing, shuffled
            removal_list = list(current_edges)
            random.shuffle(removal_list)
            
            for edge_tuple in removal_list:
                if time.time() > deadline:
                    break
                    
                if edge_tuple not in current_edges:
                    continue
                    
                # Try removing
                test_edges = current_edges - {edge_tuple}
                
                # Check if still non-2-colorable
                # Convert to list of lists for the checker
                test_list = [list(e) for e in test_edges]
                
                if not is_2_colorable_fast(n, test_list):
                    # Can remove
                    current_edges = test_edges
            
            # Final check
            final_edges = [list(e) for e in current_edges]
            if not is_2_colorable_fast(n, final_edges):
                if len(final_edges) < best_count:
                    best_count = len(final_edges)
                    best_edges = final_edges
                    best_n = n
                    print(f"Found better: n={n}, edges={best_count}")

    # Ensure we return a valid solution
    # The seed is always valid.
    return {'n': best_n, 'edges': best_edges}
