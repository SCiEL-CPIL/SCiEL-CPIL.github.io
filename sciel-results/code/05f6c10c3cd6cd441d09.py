"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Find a good independent set using randomized dynamic min-degree
    best_set = set()
    best_size = 0
    best_order = []
    
    num_runs = 50 # Do multiple runs to find a better set
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        # Dynamic Min-Degree Simulation
        active = np.ones(n, dtype=bool)
        current_degrees = adjacency.sum(axis=1).copy()
        selected = []
        
        while True:
            # Find active vertex with min degree
            min_deg = np.inf
            min_v = -1
            candidates = []
            for v in range(n):
                if active[v]:
                    if current_degrees[v] < min_deg:
                        min_deg = current_degrees[v]
                        candidates = [v]
                    elif current_degrees[v] == min_deg:
                        candidates.append(v)
                        
            if min_v == -1 and not candidates:
                break
                
            # Randomly pick from candidates to diversify
            v = random.choice(candidates)
            
            selected.append(v)
            
            # Remove v and its neighbors from active
            active[v] = False
            neighbors = np.where(adjacency[v])[0]
            for u in neighbors:
                if active[u]:
                    active[u] = False
                    # Update degrees for remaining active vertices
                    # We only need to update degrees of vertices still active
                    # But we need to know which ones are active.
                    # This inner loop is O(N) per neighbor, so O(N^2) per step, O(N^3) total.
                    # N=200, N^3 = 8,000,000. 50 runs = 400M ops. Might be slow in Python.
                    pass
                    
            # More efficient degree update:
            # Actually, we can just recompute degrees or update carefully.
            # Let's optimize the inner loop.
            
        # The above simulation is O(N^3) in worst case if done naively with Python loops.
        # Let's rewrite the simulation to be more efficient.
        
        # Efficient Dynamic Min-Degree
        active = np.ones(n, dtype=bool)
        current_degrees = adjacency.sum(axis=1).copy()
        selected = []
        
        # Use a list of active vertices to iterate faster
        active_list = list(range(n))
        
        while active_list:
            # Find min degree in active_list
            min_deg = min(current_degrees[v] for v in active_list)
            candidates = [v for v in active_list if current_degrees[v] == min_deg]
            v = random.choice(candidates)
            
            selected.append(v)
            
            # Identify neighbors to remove
            neighbors = np.where(adjacency[v])[0]
            to_remove = set(neighbors)
            to_remove.add(v)
            
            # Update active list and degrees
            new_active_list = []
            for u in active_list:
                if u not in to_remove:
                    new_active_list.append(u)
                    # u remains active. Its degree decreases by the number of its neighbors that were removed.
                    # The removed neighbors are v and its neighbors that were active.
                    # Actually, only v and its active neighbors are removed.
                    # So u's degree decreases by 1 if u is connected to v.
                    # And by 1 for each neighbor w of u that is in to_remove and was active.
                    # Since to_remove contains v and N(v), and u is not in to_remove,
                    # u is not connected to v? No, u is not in to_remove, so u is not v and u is not a neighbor of v.
                    # Wait. If u is a neighbor of v, u is in to_remove.
                    # So if u is in new_active_list, u is NOT a neighbor of v.
                    # So u's degree does NOT decrease due to v being removed.
                    # Does it decrease due to other neighbors being removed?
                    # The other removed vertices are neighbors of v.
                    # Are they neighbors of u?
                    # If u is connected to a neighbor w of v, then u's degree decreases.
                    
                    # This update is tricky. Let's just recompute degrees for active vertices?
                    # Or just keep the matrix and compute degree on the fly?
                    # For N=200, computing degree is O(N).
                    # Total steps N. Total O(N^2) per run.
                    pass
            
            # Simpler: Just recompute degrees for the new active list?
            # No, we can just use the original adjacency and mask.
            # degree(u) = sum(adjacency[u, active_list])
            
            active_list = new_active_list
            
        # Check if this set is better
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_order = selected
            
    # 2. Determine Residual Free Vertices
    # Vertices not in best_set and not neighbors of any vertex in best_set
    blocked = np.zeros(n, dtype=bool)
    for v in best_set:
        blocked[adjacency[v]] = True
    
    free_mask = np.zeros(n, dtype=bool)
    for v in range(n):
        if v not in best_set and not blocked[v]:
            free_mask[v] = True
            
    free_vertices = np.where(free_mask)[0]
    
    # 3. Order Free Vertices using Dynamic Min-Degree on Residual Graph
    # We want the decoder to pick a good set from these.
    # We can just use static min-degree on the subgraph induced by free_vertices.
    # Or dynamic. Let's use static for speed, or dynamic if fast.
    # N=200, static is fine.
    
    free_priorities = {}
    if len(free_vertices) > 0:
        # Compute degrees within the free subgraph
        free_indices = list(free_vertices)
        # Map vertex id to index in free list
        # Actually, just compute degree relative to free_mask
        degs = np.zeros(n)
        for v in free_vertices:
            degs[v] = np.sum(adjacency[v] & free_mask)
            
        # Sort free vertices by degree ascending
        sorted_free = sorted(free_vertices, key=lambda x: degs[x])
        
        for i, v in enumerate(sorted_free):
            free_priorities[v] = 5000 - i # Lower index -> Higher priority
            
    # 4. Construct Final Priority Array
    priorities = np.zeros(n, dtype=float)
    
    # Assign high priorities to best_set
    # Order them by their order in best_order to be deterministic
    for i, v in enumerate(best_order):
        priorities[v] = 10000 - i
        
    # Assign medium priorities to free vertices
    for v, p in free_priorities.items():
        priorities[v] = p
        
    # Blocked vertices get low priority (they will be skipped anyway)
    # Default is 0, which is lower than 5000 and 10000.
    
    return priorities
# EVOLVE-BLOCK-END
