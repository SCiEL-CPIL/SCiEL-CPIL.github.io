import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # We will search for the minimum number of edges
    # The best known is 51. We try to find something <= 51.
    # We'll use Simulated Annealing for a few different n values.
    
    def get_edge_mask(edge):
        mask = 0
        for v in edge:
            mask |= (1 << v)
        return mask

    def mask_to_edge(mask):
        return [v for v in range(n) if (mask >> v) & 1]

    def is_2_colorable(edge_masks, n):
        # Returns True if the hypergraph defined by edge_masks is 2-colorable.
        # Uses backtracking with unit propagation.
        
        # Pre-process: remove empty edges (none) and duplicate edges (none in set)
        # Order edges by degree of constraint? 
        # For unit propagation, we need to know for each edge which vertices are unassigned.
        
        # We represent the assignment as a list of ints: -1 (unassigned), 0, 1
        assignment = [-1] * n
        
        # Count how many edges each vertex is in? Not strictly needed for unit prop on edges directly.
        # But we need to efficiently find edges that are "close" to being monochromatic.
        
        # A simple but effective backtracking:
        # 1. Apply unit propagation until fixpoint or contradiction.
        # 2. If no unassigned vertices, check if any edge is monochromatic. If not, return True.
        # 3. Pick an unassigned vertex, try 0, recurse, try 1, recurse.
        
        # To speed up unit propagation, we can maintain a queue of "active" edges.
        # An edge is active if it has 4 or more assigned vertices.
        
        # However, implementing a full incremental constraint solver is complex.
        # Given n is small (<= 12), a simpler recursive check might suffice if optimized.
        
        # Let's use a bitmask-based check for the final verification, 
        # but for the SA loop, we need it fast.
        
        # Optimization: If the number of edges is small, the search tree is small.
        
        def propagate(assignment):
            # Returns False if contradiction, True if success.
            # Modifies assignment in place.
            changed = True
            while changed:
                changed = False
                for mask in edge_masks:
                    # Count assigned 0s and 1s in this edge
                    count0 = 0
                    count1 = 0
                    unassigned_vertex = -1
                    unassigned_count = 0
                    
                    for v in range(n):
                        if (mask >> v) & 1:
                            val = assignment[v]
                            if val == 0:
                                count0 += 1
                            elif val == 1:
                                count1 += 1
                            else:
                                unassigned_count += 1
                                unassigned_vertex = v
                                
                    if count0 == 5 or count1 == 5:
                        return False # Monochromatic edge found
                    
                    if unassigned_count == 1:
                        if count0 == 4:
                            if assignment[unassigned_vertex] != -1 and assignment[unassigned_vertex] != 1:
                                return False
                            assignment[unassigned_vertex] = 1
                            changed = True
                        elif count1 == 4:
                            if assignment[unassigned_vertex] != -1 and assignment[unassigned_vertex] != 0:
                                return False
                            assignment[unassigned_vertex] = 0
                            changed = True
            return True

        def backtrack(idx):
            # idx is index in edge_masks? No, we assign vertices.
            # This simple backtracking on vertices might be slow if n is large.
            # But n is small.
            
            # Find first unassigned vertex
            v = -1
            for i in range(n):
                if assignment[i] == -1:
                    v = i
                    break
            
            if v == -1:
                # All assigned, check if valid (propagate already checked for mono edges)
                return True
            
            # Try 0
            assignment[v] = 0
            if propagate(assignment):
                if backtrack(v + 1): # Note: this recursion depth is n
                    return True
            assignment[v] = -1
            
            # Try 1
            assignment[v] = 1
            if propagate(assignment):
                if backtrack(v + 1):
                    return True
            assignment[v] = -1
            
            return False
        
        # Initial propagation
        if not propagate(assignment):
            return False
        
        return backtrack(0)

    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try different n values
    for n in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
            
        all_edge_masks = []
        for combo in __import__('itertools').combinations(range(n), 5):
            all_edge_masks.append(get_edge_mask(list(combo)))
        
        num_edges_total = len(all_edge_masks)
        
        # Simulated Annealing
        # Objective: Minimize |E| subject to Not 2-Colorable.
        # Penalty: If Colorable, add large penalty.
        
        current_edges = set()
        # Start with a random subset of size ~45
        initial_size = 45
        random.shuffle(all_edge_masks)
        current_edges = set(all_edge_masks[:initial_size])
        
        # Check initial state
        if not is_2_colorable(list(current_edges), n):
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_n = n
                best_edges = [mask_to_edge(m) for m in current_edges]
        
        temperature = 10.0
        cooling_rate = 0.9995
        min_temp = 0.1
        
        while temperature > min_temp and time.time() < deadline - 5:
            # Propose move
            move_type = random.random()
            
            new_edges = set(current_edges)
            
            if move_type < 0.4:
                # Remove an edge
                if current_edges:
                    edge_to_remove = random.choice(list(current_edges))
                    new_edges.remove(edge_to_remove)
            elif move_type < 0.8:
                # Add an edge
                # Pick an edge not in current_edges
                # To avoid O(N) search, we can pick a random index and check
                # Or maintain a list of available edges
                # For simplicity, pick random from all, if in set, try another
                # Better: pick from a pre-shuffled list of all edges, keep a pointer?
                # Let's just pick random from all_edge_masks
                # If it's already there, do nothing or pick another
                candidates = [e for e in all_edge_masks if e not in current_edges]
                if candidates:
                    edge_to_add = random.choice(candidates)
                    new_edges.add(edge_to_add)
            else:
                # Swap
                if current_edges:
                    edge_to_remove = random.choice(list(current_edges))
                    candidates = [e for e in all_edge_masks if e not in current_edges]
                    if candidates:
                        edge_to_add = random.choice(candidates)
                        new_edges.remove(edge_to_remove)
                        new_edges.add(edge_to_add)
            
            # Calculate Objective
            # We need to check colorability of new_edges
            # This is the bottleneck.
            # To save time, we can check only if the size changed or if we suspect it.
            # Actually, we MUST check to know the penalty.
            
            # Optimization: If we removed an edge, it might become colorable.
            # If we added an edge, it might become non-colorable.
            
            new_colorable = is_2_colorable(list(new_edges), n)
            
            # Old objective
            old_colorable = is_2_colorable(list(current_edges), n) # This is expensive to recompute
            # We should cache the old colorability? 
            # In SA, we usually compute delta.
            # Here, the state space is discrete and the check is expensive.
            
            # Let's restructure: 
            # We store (edges, colorable_flag).
            # When we propose a move, we compute the new colorable_flag.
            # We keep the old one from the previous step.
            
            # Let's refactor the loop to store state properly.
            pass # The above loop is inefficient due to repeated checks.

    # Refactored SA Loop
    for n in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
            
        all_edge_masks = []
        for combo in __import__('itertools').combinations(range(n), 5):
            all_edge_masks.append(get_edge_mask(list(combo)))
        
        # Shuffle once
        random.shuffle(all_edge_masks)
        
        current_edges = set(all_edge_masks[:45])
        current_colorable = is_2_colorable(list(current_edges), n)
        
        if not current_colorable:
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_n = n
                best_edges = [mask_to_edge(m) for m in current_edges]
        
        temperature = 10.0
        cooling_rate = 0.9995
        min_temp = 0.1
        
        while temperature > min_temp and time.time() < deadline - 5:
            new_edges = set(current_edges)
            
            move_type = random.random()
            valid_move = False
            
            if move_type < 0.4 and current_edges:
                edge_to_remove = random.choice(list(current_edges))
                new_edges.remove(edge_to_remove)
                valid_move = True
            elif move_type < 0.8:
                # Add
                # Pick random edge not in current
                # To make this O(1) or O(log N), we can just pick a random index
                # and check membership. If member, skip or retry.
                # Given N is small (~1000), linear search in candidates is okay.
                # But creating candidates list is O(N).
                # Let's just pick a random edge from all_edge_masks.
                # If it's in current_edges, we might skip the move or try another.
                edge_to_add = random.choice(all_edge_masks)
                if edge_to_add not in current_edges:
                    new_edges.add(edge_to_add)
                    valid_move = True
            else:
                # Swap
                if current_edges:
                    edge_to_remove = random.choice(list(current_edges))
                    edge_to_add = random.choice(all_edge_masks)
                    if edge_to_add not in current_edges:
                        new_edges.remove(edge_to_remove)
                        new_edges.add(edge_to_add)
                        valid_move = True
            
            if not valid_move:
                continue
            
            new_colorable = is_2_colorable(list(new_edges), n)
            
            # Objective: |E| + 100 * Colorable
            old_obj = len(current_edges) + (100 if current_colorable else 0)
            new_obj = len(new_edges) + (100 if new_colorable else 0)
            
            delta = new_obj - old_obj
            
            if delta < 0 or random.random() < pow(2.718, -delta / temperature):
                current_edges = new_edges
                current_colorable = new_colorable
                
                if not current_colorable:
                    if len(current_edges) < best_count:
                        best_count = len(current_edges)
                        best_n = n
                        best_edges = [mask_to_edge(m) for m in current_edges]
            
            temperature *= cooling_rate
    
    if best_edges is None:
        # Fallback: return a known valid small graph if we didn't find one?
        # Or return the complete graph on 9 vertices? No, that's 126 edges.
        # The task says "Minimize the number of edges".
        # If we fail, we should return a valid hypergraph.
        # Let's construct a known 51-edge one? 
        # Or just return a random non-colorable one if found.
        # If nothing found, return a complete graph on 9 vertices (valid, but high score).
        # Actually, the seed program returns a result.
        # Let's ensure we return *something* valid.
        # A complete 5-uniform hypergraph on 9 vertices is not 2-colorable?
        # Yes, because any 2-coloring of 9 vertices leaves a monochromatic 5-set?
        # Ramsey number R(5,5) is unknown but > 43. So on 9 vertices, it IS 2-colorable.
        # Wait. R(3,3)=6. R(4,4)=18.
        # For 5-uniform, we need a set of 5 vertices all same color.
        # If n=9, can we 2-color it?
        # Yes, 9 vertices can be colored 4-5. No 5 monochromatic set.
        # So the complete graph on 9 vertices IS 2-colorable.
        # The seed program claims to find a non-2-colorable one.
        # Let's check the seed program logic.
        # It starts with all edges and removes them if the remaining graph is 2-colorable.
        # This implies the final graph is NOT 2-colorable.
        # So there exists a non-2-colorable 5-uniform hypergraph on 9 vertices.
        # My assumption about R(5,5) was about finding a monochromatic K5 in a graph, not a hypergraph.
        # In a hypergraph, "monochromatic edge" means all 5 vertices have the same color.
        # To force this, we need enough edges.
        
        # If best_edges is None, we need a fallback.
        # Let's just use the seed program's logic for a short time if SA fails?
        # Or hardcode a known solution?
        # The prompt says "best known construction has 51 edges".
        # I will output the best found. If none, I'll return a dummy valid one?
        # No, I must return a valid one.
        # I'll run a quick greedy removal on n=9 if SA finds nothing.
        
        if time.time() < deadline - 2:
             # Quick greedy on n=9
             n = 9
             all_edges = [list(e) for e in __import__('itertools').combinations(range(n), 5)]
             all_masks = [get_edge_mask(e) for e in all_edges]
             current = set(all_masks)
             random.shuffle(all_masks)
             for m in all_masks:
                 if time.time() > deadline - 1: break
                 if m not in current: continue
                 test = current - {m}
                 if not is_2_colorable(list(test), n):
                     current = test
             if current:
                 best_edges = [mask_to_edge(m) for m in current]
                 best_n = n

    if best_edges is None:
        # Absolute fallback: return empty? No, must be valid.
        # Return a single edge? A single edge IS 2-colorable.
        # We need a non-2-colorable one.
        # If we can't find one, we fail.
        # But SA should find one.
        pass

    return {'n': best_n if best_n else 9, 'edges': best_edges if best_edges else []}

