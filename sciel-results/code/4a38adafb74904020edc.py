import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate edges until we have enough or tried enough
        # The seed program tries 4*n*d times. Let's stick to similar density logic.
        target_edges = n * d // 2 # Approx
        attempts = 0
        max_attempts = 10 * n * d
        while len(edges) < target_edges and attempts < max_attempts:
            attempts += 1
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            e = (u, v)
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, time_limit):
        if not edges:
            return []
        
        n_edges = len(edges)
        # Map edge to index
        edge_index = {e: i for i, e in enumerate(edges)}
        
        # Build conflict sets
        # conflict[i] is a set of indices j such that edge i and edge j conflict
        conflict = [set() for _ in range(n_edges)]
        
        # Helper to check if two edges conflict
        # Two edges (u,v) and (x,y) conflict if:
        # 1. They share a vertex (u==x, u==y, v==x, v==y)
        # 2. They are connected by an edge in the original graph.
        #    i.e., (u,x) in edges, (u,y) in edges, (v,x) in edges, or (v,y) in edges.
        
        # To speed this up, we can precompute adjacency of vertices.
        # But here we are dealing with edge-edge conflicts.
        # Let's just iterate pairs. n_edges is at most n*d/2.
        # For n=80, d=6, n_edges <= 240. 240^2 = 57600 pairs. Very fast.
        
        edge_set = set(edges)
        
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check vertex sharing
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue # No need to check adjacency if they share vertex? 
                             # Actually, the definition says "share a vertex OR are joined by an edge".
                             # If they share a vertex, they conflict. The "joined by an edge" part is for disjoint edges.
                             # So if they share a vertex, they conflict.
                             # If they don't share a vertex, check if (u1,u2), (u1,v2), (v1,u2), (v1,v2) are in edge_set.
                else:
                    if (u1, u2) in edge_set or (u1, v2) in edge_set or (v1, u2) in edge_set or (v1, v2) in edge_set:
                        # Ensure tuple order for set lookup
                        # edge_set contains (min, max)
                        def in_edges(a, b):
                            return (min(a,b), max(a,b)) in edge_set
                        if in_edges(u1, u2) or in_edges(u1, v2) or in_edges(v1, u2) or in_edges(v1, v2):
                            conflict[i].add(j)
                            conflict[j].add(i)

        # Greedy Initial Coloring
        # Order edges by degree in conflict graph (highest first) to improve greedy result
        order = sorted(range(n_edges), key=lambda x: len(conflict[x]), reverse=True)
        
        colours = [None] * n_edges
        used_colours = set()
        
        for idx in order:
            # Find available colour
            # A colour c is invalid if any neighbour j of idx has colours[j] == c
            invalid = set()
            for j in conflict[idx]:
                if colours[j] is not None:
                    invalid.add(colours[j])
            
            # Pick smallest available colour
            c = 0
            while c in invalid:
                c += 1
            colours[idx] = c
            used_colours.add(c)
        
        current_num_colours = len(used_colours)
        best_colours = colours.copy()
        best_num_colours = current_num_colours
        
        rng = random.Random(42)
        
        def try_remove_colour(c):
            # Try to recolor all edges of colour c to other colours
            # Returns True if successful, and updates colours if so
            # We need to be careful with state management
            edges_of_c = [i for i in range(n_edges) if colours[i] == c]
            if not edges_of_c:
                return True
            
            # Save state
            original_colours = [colours[i] for i in edges_of_c]
            
            # Try to recolor
            # We process edges one by one.
            # This is a simplified backtracking. If we get stuck, we revert.
            # For better performance, we can use a more sophisticated algorithm, 
            # but given the small size, a simple greedy-with-backtrack or even just greedy 
            # might work if we order the edges of c smartly.
            
            # Let's try a recursive backtracking for the recoloring of edges_of_c
            # State: which edges have been recolored
            # To keep it fast, we limit the depth or use a heuristic.
            
            # Heuristic: Recolor edges of c in order of increasing available options
            
            def get_available_colours(idx, current_assignments):
                # current_assignments is a dict idx -> colour for edges being modified
                # Other edges retain their colours in 'colours'
                invalid = set()
                for j in conflict[idx]:
                    if j in current_assignments:
                        invalid.add(current_assignments[j])
                    else:
                        if colours[j] != c: # If neighbour is not c, its colour is fixed
                            invalid.add(colours[j])
                        # If neighbour is c, it is in current_assignments or will be.
                        # If it's c and not assigned yet, it's not a constraint yet?
                        # No, if neighbour is c, and we are trying to remove c, the neighbour will also be changed.
                        # So we only care about neighbours that are NOT c in the final state.
                        # But during the process, we don't know the final state.
                        # This makes simple backtracking tricky.
                        # Alternative: Just try to change to any colour != c that is valid w.r.t current state.
                        # This is what we'll do. It's not perfect but often works.
                        pass 
                
                # Simpler check: 
                # A colour k is valid for idx if no neighbour j has colour k in the *current* state 
                # where j is not being changed, OR j is being changed to k? No, if j is changed to k, that's fine?
                # No, if j is changed to k, and i is changed to k, and i,j conflict, that's bad.
                # So we must ensure that if j is in the set of edges being changed, its *new* colour is not k.
                
                # This is getting complicated. Let's use a simpler local search move:
                # Swap colours between two edges?
                # Or just random recoloring of a single edge?
                
                pass

            # Let's implement a simpler, effective local search:
            # 1. Pick a random edge e.
            # 2. Try to change its colour to a random other valid colour.
            # 3. Accept if it reduces the number of colours, or with probability p if it keeps it same.
            # 4. If it increases, reject.
            
            # This is Simulated Annealing / Random Walk.
            # To specifically target reducing colour count, we bias towards changing colours 
            # that are used by only one edge (unique colours) to merge them.
            
            return False # Placeholder

        # Implementing a robust SA is better.
        # State: colours array
        # Cost: number of distinct colours
        # Move: Pick random edge i, pick random colour c' != colours[i].
        #        If c' is valid for i (no conflict with neighbours), accept.
        #        This maintains validity.
        #        The cost (num colours) can go up or down.
        #        If it goes down, great. If up, accept with prob exp(-delta/T).
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        time_limit_instance = time_limit - (time.time() - start_time)
        if time_limit_instance <= 0:
            return best_colours
            
        steps = 0
        while time.time() < start_time + time_limit_instance:
            steps += 1
            # Pick random edge
            i = rng.randrange(n_edges)
            old_c = colours[i]
            
            # Pick new colour
            # To encourage merging, pick from existing colours mostly
            # If we pick a new colour, cost increases.
            # Let's pick from used_colours
            if not used_colours:
                break
                
            # Randomly choose a target colour from used_colours
            # Or a new one? 
            # If we want to minimize, we should try to change to an existing colour.
            # But if we always do that, we might get stuck.
            # Let's pick a random integer 0..max(used_colours)+1
            
            max_c = max(used_colours) if used_colours else 0
            new_c = rng.randint(0, max_c + 1)
            
            if new_c == old_c:
                continue
                
            # Check validity
            valid = True
            for j in conflict[i]:
                if colours[j] == new_c:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Calculate cost change
            # If old_c is unique, removing it decreases cost by 1.
            # If new_c is new (not in used_colours), adding it increases cost by 1.
            # Otherwise, cost stays same.
            
            count_old = sum(1 for c in colours if c == old_c)
            count_new = sum(1 for c in colours if c == new_c)
            
            delta_cost = 0
            if count_old == 1:
                delta_cost -= 1
            if new_c not in used_colours:
                delta_cost += 1
                
            # Acceptance
            if delta_cost < 0:
                # Improvement
                colours[i] = new_c
                if count_old == 1:
                    used_colours.remove(old_c)
                if new_c not in used_colours:
                    used_colours.add(new_c)
                current_num_colours += delta_cost
                if current_num_colours < best_num_colours:
                    best_num_colours = current_num_colours
                    best_colours = colours.copy()
            else:
                # Same or worse
                if delta_cost == 0:
                    # Accept with some probability to explore
                    if rng.random() < 0.5:
                        colours[i] = new_c
                        # used_colours set doesn't change membership, but we should verify
                        # Actually, if count_old > 1 and new_c in used, set is unchanged.
                        # If count_old > 1 and new_c not in used, set grows. (But we said delta_cost 0, so new_c must be in used)
                        # So set is unchanged.
                else:
                    # Worse, accept with prob exp(-delta/T)
                    prob = 2.71828 ** (-delta_cost / T)
                    if rng.random() < prob:
                        colours[i] = new_c
                        if new_c not in used_colours:
                            used_colours.add(new_c)
                        # old_c is still used by others (count_old > 1)
                        current_num_colours += delta_cost
                        
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        return best_colours

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Not enough time, return empty or best effort? 
            # Better to return valid solution. If we can't solve, return empty list? 
            # The task implies we must return a list. 
            # If we run out of time, we should return the best we have.
            # But we haven't started. 
            # Let's just break and hope the previous ones are done.
            break
            
        edges = instance(*spec)
        # Allocate time per instance
        remaining = deadline - time.time()
        # Reserve some time for the next ones?
        # Simple strategy: give each instance 1/3 of remaining time, but at least a minimum.
        # Or just run until deadline - margin.
        # Since there are 3 instances, let's give each ~1/3 of the available time.
        time_limit = (deadline - time.time()) / (3 - specs.index(spec))
        
        sol = solve_instance(edges, time_limit)
        out.append(sol)
        
    return out

if __name__ == "__main__":
    construct()
