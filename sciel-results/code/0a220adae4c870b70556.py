"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for speed
    adj_mask = [0] * n
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    def find_hamiltonian_cycle(adj_masks, start_node=0, deadline=None, max_steps=500000):
        """
        Finds a Hamiltonian cycle using bitmask-based DFS with pruning.
        adj_masks: list of int bitmasks where bit i is set if vertex i is adjacent
        Returns a list of vertices representing the cycle, or None if not found.
        """
        if deadline is None:
            deadline = start_time + 85.0
        
        n_local = len(adj_masks)
        full_mask = (1 << n_local) - 1
        start_mask = 1 << start_node
        
        # Pre-check: all vertices must have degree >= 2
        for u in range(n_local):
            if bin(adj_masks[u]).count('1') < 2:
                return None
        
        steps = [0]
        path = [start_node]
        
        def backtrack():
            if time.time() > deadline or steps[0] > max_steps:
                return False
            
            steps[0] += 1
            current = path[-1]
            
            if len(path) == n_local:
                if (adj_masks[current] >> start_node) & 1:
                    return True
                return False
            
            # Get unvisited neighbors from bitmask
            visited_mask = 0
            for v in path:
                visited_mask |= (1 << v)
            
            # Unvisited neighbors of current
            candidates = adj_masks[current] & ~visited_mask
            
            if candidates == 0:
                return False
            
            # Build list of candidate vertices
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cand_list.append(v)
                temp ^= lsb
            
            # Sort by degree in remaining subgraph (most constrained first)
            def priority(v):
                # Count unvisited neighbors of v
                deg = 0
                t = adj_masks[v] & ~visited_mask
                while t:
                    t &= t - 1
                    deg += 1
                return deg
            
            cand_list.sort(key=priority)
            
            for next_node in cand_list:
                # Pruning: if this is the last vertex, it must connect to start
                if len(path) + 1 == n_local:
                    if not ((adj_masks[next_node] >> start_node) & 1):
                        continue
                
                # Connectivity pruning: check if all unvisited vertices remain connected
                # to the path or to each other
                if len(path) > n_local // 3:
                    new_visited = visited_mask | (1 << next_node)
                    # Check each unvisited vertex has at least one neighbor in 
                    # (path ∪ {next_node}) or another unvisited vertex
                    unvisited = full_mask & ~new_visited
                    if unvisited:
                        # Simple check: for each unvisited vertex, it must have a neighbor
                        # that is either in the path, is next_node, or is another unvisited vertex
                        stuck = False
                        t = unvisited
                        while t:
                            lsb = t & (-t)
                            u = lsb.bit_length() - 1
                            t ^= lsb
                            # Neighbors of u that are in path or next_node or unvisited
                            # (i.e., not completely isolated from the growing path)
                            # u must have at least one neighbor in (path ∪ {next_node}) 
                            # OR at least one other unvisited neighbor
                            neighbors_in_path = adj_masks[u] & (new_visited)
                            if neighbors_in_path == 0:
                                # Must have at least one unvisited neighbor
                                other_unvisited = adj_masks[u] & unvisited
                                if other_unvisited == 0:
                                    stuck = True
                                    break
                        if stuck:
                            continue
                
                path.append(next_node)
                if backtrack():
                    return True
                path.pop()
            
            return False
        
        if backtrack():
            return path[:]
        return None
    
    def try_find_cycles(adj_masks, max_cycles=3, deadline=None, random_seed=None):
        """Try to find max_cycles edge-disjoint Hamiltonian cycles."""
        if deadline is None:
            deadline = start_time + 85.0
        
        cycles = []
        current_masks = adj_masks[:]
        
        for i in range(max_cycles):
            if time.time() > deadline:
                break
            
            # Check viability
            valid = True
            for u in range(n):
                if bin(current_masks[u]).count('1') < 2:
                    valid = False
                    break
            if not valid:
                break
            
            # Try multiple starting nodes
            cycle = None
            starts = list(range(n))
            if random_seed is not None:
                random.seed(random_seed + i)
                random.shuffle(starts)
            
            for start in starts:
                if time.time() > deadline:
                    break
                cycle = find_hamiltonian_cycle(current_masks, start_node=start, deadline=deadline)
                if cycle is not None:
                    break
            
            if cycle is None:
                break
            
            cycles.append(cycle)
            
            # Remove edges of this cycle
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_masks[u] &= ~(1 << v)
                current_masks[v] &= ~(1 << u)
        
        return cycles
    
    # First: deterministic attempt
    best_cycles = try_find_cycles(adj_mask, max_cycles=3, deadline=deadline)
    
    # If we didn't get 3, try random restarts
    if len(best_cycles) < 3:
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 3.0:
                break
            
            seed = 1000 + attempt
            cycles = try_find_cycles(adj_mask, max_cycles=3, deadline=deadline, random_seed=seed)
            
            if len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) == 3:
                    break
    
    # Fallback: ensure we return at least one valid cycle
    if not best_cycles:
        # Try with different max_steps
        for ms in [1000000, 2000000]:
            for start in range(n):
                if time.time() > deadline - 1.0:
                    break
                cycle = find_hamiltonian_cycle(adj_mask, start_node=start, deadline=deadline, max_steps=ms)
                if cycle is not None:
                    best_cycles = [cycle]
                    break
            if best_cycles:
                break
        
        if not best_cycles:
            # Minimal fallback - find any cycle
            best_cycles = [[0, 1, 2]]
            valid = True
            for i in range(3):
                u = best_cycles[0][i]
                v = best_cycles[0][(i + 1) % 3]
                if not ((adj_mask[u] >> v) & 1):
                    valid = False
                    break
            if not valid:
                best_cycles = []
    
    # Convert to list of lists of Python ints
    result = [list(c) for c in best_cycles]
    
    return result
# EVOLVE-BLOCK-END
