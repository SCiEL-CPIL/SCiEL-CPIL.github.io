import time
import random
from itertools import combinations

def check_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses optimized backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Build incidence structure
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    edge_c0 = [0] * len(edges)
    edge_c1 = [0] * len(edges)
    
    def backtrack():
        # 1. Unit Propagation
        forced = True
        while forced:
            forced = False
            for i in range(len(edges)):
                c0 = edge_c0[i]
                c1 = edge_c1[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            for j in vertex_to_edges[v]:
                                edge_c1[j] += 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            for j in vertex_to_edges[v]:
                                edge_c0[j] += 1
                            forced = True
                            break
                            
        # Check for contradictions
        for i in range(len(edges)):
            if edge_c0[i] == 5 or edge_c1[i] == 5:
                return False
        
        # Find unassigned vertex
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
                
        if v == -1:
            return True
        
        # Save state
        colors_copy = colors[:]
        c0_copy = edge_c0[:]
        c1_copy = edge_c1[:]
        
        # Try 0
        colors[v] = 0
        for j in vertex_to_edges[v]:
            edge_c0[j] += 1
            
        if backtrack():
            return True
            
        # Restore
        colors[:] = colors_copy
        edge_c0[:] = c0_copy
        edge_c1[:] = c1_copy
        
        # Try 1
        colors[v] = 1
        for j in vertex_to_edges[v]:
            edge_c1[j] += 1
            
        if backtrack():
            return True
            
        # Restore
        colors[:] = colors_copy
        edge_c0[:] = c0_copy
        edge_c1[:] = c1_copy
        
        return False

    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
            
        # Start with complete hypergraph
        all_edges = [list(e) for e in combinations(range(n), 5)]
        
        # Check if complete is non-2-colorable
        if not check_non_2_colorable(n, all_edges):
            continue
            
        current_edges = all_edges[:]
        current_m = len(current_edges)
        
        # Randomize order of removal
        indices = list(range(current_m))
        random.shuffle(indices)
        
        # Try to remove edges
        # We iterate through the shuffled indices
        # If removing an edge keeps it non-2-colorable, we remove it.
        # This is a greedy approach.
        
        # To improve, we can do multiple passes or restarts.
        # Given time constraints, one pass per n is a good start.
        
        # Optimization: If current_m is already >= best_m, we might still find a smaller one?
        # No, we are removing edges, so m decreases.
        # If current_m < best_m, we have a new best.
        
        # We should check if we can beat the current best.
        # If best_m is 16, and we start with 126, we will eventually check m=15.
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            if current_m == 0:
                break
                
            # If current_m is already worse than best_m and we are removing, 
            # we might skip checks if we know we can't beat best_m?
            # No, we need to know if it's valid.
            
            # Try removing edge at idx
            # Note: indices are based on original list, but list shrinks.
            # So we need to track which edges are still present.
            # Simpler: keep a set of active edges.
            # But copying lists is fast for n<=14.
            
            # Let's use a different strategy for removal:
            # Maintain a list of active edges.
            # At each step, pick a random active edge to try removing.
            
            # Actually, the loop above `for idx in indices` is flawed because 
            # `current_edges` changes size.
            # Let's restart the removal logic properly.
            
            break # Break out of the flawed loop structure and use a while loop
        
        # Correct Removal Loop
        active_indices = set(range(len(all_edges)))
        current_edges_list = [all_edges[i] for i in active_indices]
        
        # We will try to remove edges one by one.
        # We need a list of candidate indices to try.
        # We'll just pick random indices from the current list.
        
        # To avoid infinite loops or getting stuck, we limit the number of failed attempts.
        failed_attempts = 0
        max_failed = 50
        
        while time.time() < deadline - 5:
            if not active_indices:
                break
                
            # Pick a random active edge to remove
            # Convert set to list for random choice
            active_list = list(active_indices)
            idx_to_try = random.choice(active_list)
            
            # Try removing it
            new_active = active_indices - {idx_to_try}
            new_edges = [all_edges[i] for i in new_active]
            
            if check_non_2_colorable(n, new_edges):
                # Success
                active_indices = new_active
                current_edges_list = new_edges
                failed_attempts = 0
                
                if len(new_edges) < best_m:
                    best_m = len(new_edges)
                    best_edges = new_edges[:]
                    best_n = n
                    
            else:
                failed_attempts += 1
                if failed_attempts > max_failed:
                    # Stuck, try a different n or restart?
                    # For now, break to try next n
                    break

    # Final verification
    if best_edges is not None:
        if check_non_2_colorable(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a known small one if found, else empty
    # If no valid one found (shouldn't happen for n>=9), return empty
    return {'n': best_n if best_edges else 0, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    if res['edges']:
        assert not check_non_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
