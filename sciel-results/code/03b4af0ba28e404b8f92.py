import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Diagonal constraint: 2a is a perfect square
    # a = 2 * m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
        
    # Available vertices
    available_vertices = [i for i in range(1, n + 1) if i not in diagonal_bad]
    available_set = set(available_vertices)
    
    # Build adjacency lists (excluding diagonal_bad vertices from neighbors? 
    # No, keep full graph, just don't pick diagonal_bad vertices)
    # conflicts[v] = set of u such that u+v is square and u != v
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Greedy Initialization using Minimum Degree Heuristic
    def greedy_init():
        current_set = set()
        remaining = set(available_vertices)
        
        while remaining:
            # Find vertex with minimum degree in the induced subgraph of 'remaining'
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            current_set.add(best_v)
            # Remove best_v and its neighbors from remaining
            to_remove = {best_v} | (conflicts[best_v] & remaining)
            remaining -= to_remove
            
        return current_set

    # Local Search
    def local_search(s, time_limit):
        current = set(s)
        in_set = set(current)
        
        # Precompute for faster neighbor checks? 
        # We will just use set intersections.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            # We can iterate through available vertices not in set
            for v in available_vertices:
                if v in in_set:
                    continue
                # Check if v has any neighbor in current
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    break # Restart loop to check new state
            
            if not improved:
                # 2. Try 1-for-2 swap
                # For each u in current, find candidates v not in current such that 
                # conflicts[v] & current == {u}
                # If we find two such v1, v2 that are not adjacent, swap u -> {v1, v2}
                
                # To make this efficient, we can precompute for each u the list of "blocked only by u" vertices
                # But this list changes as we add/remove.
                # Given N=256, we can just compute on the fly for each u in current.
                
                swap_done = False
                for u in list(current):
                    candidates = []
                    for v in available_vertices:
                        if v in in_set:
                            continue
                        # v is blocked by current if (conflicts[v] & in_set) is non-empty
                        # We want (conflicts[v] & in_set) to be exactly {u}
                        # This implies (conflicts[v] & in_set) is not empty, and 
                        # (conflicts[v] & (in_set - {u})) is empty
                        # So conflicts[v] & in_set == {u}
                        
                        # Optimization: 
                        # 1. Check if u is in conflicts[v]. If not, skip.
                        if u not in conflicts[v]:
                            continue
                        # 2. Check if any other neighbor of v is in in_set (excluding u)
                        # i.e., (conflicts[v] & in_set) - {u} is empty
                        # Equivalent to: (conflicts[v] & in_set) == {u}
                        
                        # Fast check:
                        # If len(conflicts[v] & in_set) > 1, skip
                        # If len(conflicts[v] & in_set) == 1, it must be {u} because u is in it.
                        
                        common = conflicts[v] & in_set
                        if len(common) == 1 and u in common:
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        # Find two non-adjacent candidates
                        found_swap = False
                        for i in range(len(candidates)):
                            if found_swap: break
                            for j in range(i + 1, len(candidates)):
                                if candidates[i] not in conflicts[candidates[j]]:
                                    # Swap!
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates[i])
                                    in_set.add(candidates[i])
                                    current.add(candidates[j])
                                    in_set.add(candidates[j])
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    break
                        if swap_done:
                            break # Restart outer loop
                
                if not swap_done:
                    # 3. Perturbation: Remove 1-3 random vertices and refill
                    # This helps escape local minima
                    if time.time() > time_limit - 2:
                        break
                        
                    # Random removal
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    for v in to_remove:
                        current.remove(v)
                        in_set.remove(v)
                    
                    # Refill greedily
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in available_vertices:
                            if v in in_set:
                                continue
                            if not (conflicts[v] & in_set):
                                current.add(v)
                                in_set.add(v)
                                refill_improved = True
                                break
                                
                    # If size increased, we are improved. If not, we still accept 
                    # the perturbation to allow further search? 
                    # Usually, we accept if size >= previous best, or with probability.
                    # For simplicity, if size didn't increase, we might have lost ground.
                    # But perturbation is meant to jump out. 
                    # Let's only mark improved if size increased or stayed same?
                    # Actually, if we just want to find *a* good solution, 
                    # we should track the best solution seen so far.
                    pass

        return current

    best_set = set()
    best_size = 0
    
    # Initial greedy
    s = greedy_init()
    best_set = set(s)
    best_size = len(s)
    
    # Local search on initial
    s = local_search(s, deadline - 5)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
        
    # Multiple restarts
    random.seed(42)
    
    while time.time() < deadline - 2:
        # Randomized Greedy Init
        # Shuffle available, then pick min degree among remaining?
        # Or just random order greedy?
        # Let's try: Random permutation, then greedy add if possible.
        order = available_vertices[:]
        random.shuffle(order)
        
        current = set()
        in_set = set()
        for v in order:
            if not (conflicts[v] & in_set):
                current.add(v)
                in_set.add(v)
                
        # Local search
        current = local_search(current, deadline - 2)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
    return list(best_set)
