import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 26
    all_edges = list(itertools.combinations(range(n), 5))
    
    # Fast check for 2-colorability using backtracking
    def is_not_2_colorable(edges):
        if not edges:
            return False
            
        # colors: -1 unassigned, 0, 1
        colors = [-1] * n
        
        # Precompute edge indices for faster access?
        # For n=26, edges are small.
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    unassigned = []
                    for v in e:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unassigned.append(v)
                            
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and unassigned:
                        for v in unassigned:
                            colors[v] = 1
                            changed = True
                    elif c1 == 4 and unassigned:
                        for v in unassigned:
                            colors[v] = 0
                            changed = True
            return True

        def backtrack(idx):
            # Find next unassigned
            while idx < n and colors[idx] != -1:
                idx += 1
            if idx == n:
                return True
                
            # Heuristic: try the color that is less constrained?
            # For now, try 0 then 1
            
            # Try 0
            colors[idx] = 0
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
            # Try 1
            colors[idx] = 1
            if propagate():
                if backtrack(idx + 1):
                    return True
            colors[idx] = -1
            
            return False

        return not backtrack(0)

    best_edges = None
    best_count = float('inf')
    
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Initialize random set of edges
        # Start with a moderate number of edges
        # If we start too sparse, it might be 2-colorable.
        # If we start too dense, it's likely not, but minimization takes longer.
        
        initial_size = 120
        current_edges = random.sample(all_edges, initial_size)
        
        # Ensure it is not 2-colorable
        # If it is 2-colorable, add edges until it is not
        while is_not_2_colorable(current_edges) == False:
            if time.time() > deadline - 5:
                break
            # Add a random edge not in current set
            # To speed up, we can just add random edges from all_edges
            # and check less frequently?
            # For now, add one and check.
            # Optimization: If we add many, check every 5.
            for _ in range(5):
                current_edges.append(random.choice(all_edges))
            # Remove duplicates
            current_edges = list(set(current_edges))
            
            if len(current_edges) > 200:
                break
                
        if is_not_2_colorable(current_edges):
            # Minimize using Simulated Annealing / Hill Climbing
            # We try to remove edges.
            # If removal keeps it not 2-colorable, accept.
            # If removal makes it 2-colorable, reject (or accept with prob if we allow temporary violations? No, constraint is hard).
            # So it's a greedy removal with random restarts/shuffling.
            
            # Shuffle current edges
            random.shuffle(current_edges)
            
            # Greedy removal
            new_edges = []
            for e in current_edges:
                # Try to remove e
                # Candidate: new_edges (without e)
                # If new_edges is not 2-colorable, e is redundant.
                # If new_edges is 2-colorable, e is needed.
                
                # To speed up, we can check is_not_2_colorable(new_edges)
                # This is O(len(new_edges) * backtrack_cost)
                
                if is_not_2_colorable(new_edges):
                    # e is redundant
                    pass
                else:
                    new_edges.append(e)
                    
            if len(new_edges) < best_count:
                best_count = len(new_edges)
                best_edges = new_edges[:]
                
            # If we have time, do a second pass or SA?
            # For now, one greedy pass per trial is a structural improvement over the seed's single pass on full graph.
            # The seed did 50 trials of greedy on full graph.
            # We do 20 trials of greedy on random sparse graph.
            # This explores different basins.

    if best_edges is None:
        # Fallback: return a known valid small hypergraph if possible?
        # Or return the seed's result?
        # We must return a valid hypergraph.
        # If we found nothing, we might have a timeout issue.
        # Let's return a dense graph on n=9 which is definitely not 2-colorable.
        # Actually, the seed program returns a valid one.
        # We should ensure we always return a valid one.
        
        # Fallback: Complete graph on n=9 (126 edges)
        n_fallback = 9
        edges_fallback = list(itertools.combinations(range(n_fallback), 5))
        return {'n': n_fallback, 'edges': [list(e) for e in edges_fallback]}
        
    return {'n': n, 'edges': [list(e) for e in best_edges]}
