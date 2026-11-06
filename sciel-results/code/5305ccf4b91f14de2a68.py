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
        
    # Pre-compute edge indices for faster access? 
    # For n <= 26, list iteration is fine.
    
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
                unassigned_count = 0
                for v in e:
                    cv = assign[v]
                    if cv == 0:
                        c0 += 1
                    elif cv == 1:
                        c1 += 1
                    else:
                        unassigned_count += 1
                        if unassigned == -1:
                            unassigned = v
                        
                if c0 == 5 or c1 == 5:
                    return False
                    
                if c0 == 4 and unassigned_count == 1:
                    assign[unassigned] = 1
                    changed = True
                elif c1 == 4 and unassigned_count == 1:
                    assign[unassigned] = 0
                    changed = True
        return True

    def solve(assign):
        # Find next unassigned vertex
        # Heuristic: Pick vertex that appears in the most "critical" edges
        # Critical edge: 4 assigned, 1 unassigned.
        
        best_v = -1
        max_score = -1
        
        # Calculate score for each unassigned vertex
        # Score = number of edges where this vertex is the only unassigned one (and 4 are assigned)
        # If no such edges, score = degree of vertex (number of edges containing it)
        
        # To avoid O(n * E) every step, we can just pick the first unassigned if we are fast enough.
        # But for hard instances, heuristic matters.
        
        # Simple heuristic: First unassigned vertex.
        # Let's try a slightly better one: Vertex with highest degree among unassigned.
        
        # Pre-compute degrees? No, dynamic.
        
        unassigned_vertices = [i for i in range(n) if assign[i] == -1]
        if not unassigned_vertices:
            return True
            
        # Pick vertex with highest "constraint"
        # Approximation: Count how many edges have 4 assigned vertices and this unassigned vertex.
        
        # This is O(E * 5) per step, which is fine for n=26, E~50.
        
        scores = {v: 0 for v in unassigned_vertices}
        for e in edges:
            c0 = 0
            c1 = 0
            unassigned_in_e = []
            for v in e:
                cv = assign[v]
                if cv == 0:
                    c0 += 1
                elif cv == 1:
                    c1 += 1
                else:
                    unassigned_in_e.append(v)
                    
            if len(unassigned_in_e) == 1:
                v = unassigned_in_e[0]
                if v in scores:
                    scores[v] += 10 # High priority for critical edges
            
            # Also add a small weight for general degree
            # This part is expensive if done every time. 
            # Let's just use the critical edge count.
            
        if max(scores.values()) > 0:
            # Pick one with max score
            best_v = max(scores, key=scores.get)
        else:
            # Pick random unassigned or first
            best_v = random.choice(unassigned_vertices)
            
        v = best_v
        
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
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # We know m(5) <= 51. 
    # Let's search n from 9 to 26.
    # For each n, we try to find a small non-2-colorable hypergraph.
    
    for n in range(9, 27):
        if time.time() > deadline:
            break
            
        # Number of restarts
        # Allocate time based on n. Smaller n is easier to solve but harder to minimize?
        # Actually, larger n allows for sparser hypergraphs.
        # The known 51-edge construction is on n=20 or so? 
        # Let's just do a fixed number of restarts, say 10-20, depending on time.
        
        restarts = 10
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a random subset
            # Heuristic: Start with ~50-70 edges
            k = min(len(all_possible), 50 + random.randint(0, 30))
            edges = [list(e) for e in all_possible[:k]]
            
            # Ensure it's non-2-colorable
            # If it is, add random edges until it's not
            # Use a set for fast lookup
            edge_set = set(tuple(e) for e in edges)
            
            while is_2_colorable_fast(n, edges):
                if time.time() > deadline:
                    break
                if len(edges) < len(all_possible):
                    # Add a random edge not in edges
                    # To be efficient, just pick from the remaining all_possible
                    # Since we shuffled, we can keep an index
                    # Simpler: pick random from all_possible, check if in set
                    # This is O(1) average
                    while True:
                        new_edge = tuple(random.choice(all_possible))
                        if new_edge not in edge_set:
                            edges.append(list(new_edge))
                            edge_set.add(new_edge)
                            break
                else:
                    break
            
            if time.time() > deadline:
                break
                
            # Verify it is non-2-colorable
            if is_2_colorable_fast(n, edges):
                continue
                
            # Greedy deletion
            # We want to remove edges one by one if possible.
            current_edges = set(tuple(e) for e in edges)
            
            # Multiple passes of deletion
            # In each pass, try to remove each edge.
            # If successful, the set changes, so we restart the pass?
            # Or just iterate through the original list, checking if still present.
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Create a list of current edges to iterate over
                # Shuffling helps avoid bias
                removal_list = list(current_edges)
                random.shuffle(removal_list)
                
                for edge_tuple in removal_list:
                    if time.time() > deadline:
                        break
                        
                    if edge_tuple not in current_edges:
                        continue
                        
                    # Try removing
                    test_edges = current_edges - {edge_tuple}
                    test_list = [list(e) for e in test_edges]
                    
                    if not is_2_colorable_fast(n, test_list):
                        # Can remove
                        current_edges = test_edges
                        improved = True
                        # If we removed one, we could restart the loop immediately
                        # to take advantage of the new state
                        # But iterating through the rest of the current list is also fine.
                        # Restarting might be better to catch new opportunities.
                        # Let's break and restart the while loop
                        break
            
            # Final check
            final_edges = [list(e) for e in current_edges]
            if not is_2_colorable_fast(n, final_edges):
                if len(final_edges) < best_count:
                    best_count = len(final_edges)
                    best_edges = final_edges
                    best_n = n
                    print(f"Found better: n={n}, edges={best_count}")

    # If no better found, return a known valid one (Complete graph on 9)
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
