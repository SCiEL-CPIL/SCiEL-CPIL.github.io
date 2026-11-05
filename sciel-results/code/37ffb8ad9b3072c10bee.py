import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflicts using bitsets for speed
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))  # bit (j-1) represents vertex j
        conflict_mask[i] = mask
    
    # Helper to get neighbors in remaining set
    def get_neighbors(v, remaining_mask):
        """Returns bitmask of neighbors of v that are in remaining_mask."""
        return conflict_mask[v] & remaining_mask
    
    # Greedy MIS with min-degree heuristic
    def greedy_mis(vertices_list, remaining_mask, min_degree_order=True):
        """
        Greedy independent set.
        vertices_list: list of vertices to consider
        remaining_mask: bitmask of available vertices
        Returns: (solution_list, final_remaining_mask)
        """
        solution = []
        # Work with a mutable copy of remaining
        rem = remaining_mask
        
        while rem:
            if min_degree_order:
                # Find vertex with minimum degree in the induced subgraph on rem
                min_v = None
                min_deg = float('inf')
                # Iterate through bits set in rem
                temp = rem
                while temp:
                    # Get lowest set bit
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1  # 0-indexed
                    v = v_idx + 1  # vertex number
                    
                    # Count neighbors in rem
                    deg = (conflict_mask[v] & rem).bit_count() - 1  # exclude self? No, conflict_mask doesn't have self
                    # Actually conflict_mask[v] doesn't include v itself, so:
                    deg = (conflict_mask[v] & rem).bit_count()
                    
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb  # clear the bit
            else:
                # Just pick the lowest vertex
                lsb = rem & (-rem)
                min_v = lsb.bit_length()  # vertex number
                
            solution.append(min_v)
            
            # Remove min_v and its neighbors from rem
            neighbors = conflict_mask[min_v]
            rem &= ~(1 << (min_v - 1))  # remove min_v
            rem &= ~neighbors  # remove neighbors
            
        return solution, rem
    
    # Initial greedy solution
    all_mask = (1 << N) - 1  # bits 0..255 set
    best_sol, _ = greedy_mis(vertices, all_mask, min_degree_order=True)
    best_size = len(best_sol)
    best_sol_set = set(best_sol)
    best_mask = 0
    for v in best_sol:
        best_mask |= (1 << (v - 1))
    
    # Fast polish: add any vertex that doesn't conflict with current set
    def polish(sol_mask):
        """Add all vertices that don't conflict with sol_mask. Returns (new_mask, added_count)."""
        added = 0
        m = sol_mask
        for v in vertices:
            if m & (1 << (v - 1)):
                continue
            # Check if v conflicts with any vertex in m
            if (conflict_mask[v] & m) == 0:
                m |= (1 << (v - 1))
                added += 1
        return m, added
    
    # Apply initial polish
    best_mask, added = polish(best_mask)
    if added > 0:
        best_sol = [v for v in vertices if best_mask & (1 << (v - 1))]
        best_size = len(best_sol)
        best_sol_set = set(best_sol)
    
    # Randomized restarts with different tie-breaking
    best_sol = list(best_sol)
    
    for iteration in range(12000):  # Max restarts
        if time.time() > deadline:
            break
        
        # Randomized greedy: pick from top-k min-degree vertices
        rem = all_mask
        sol = []
        
        while rem:
            # Find candidates with low degree
            temp = rem
            candidates = []
            deg_list = []
            
            while temp:
                lsb = temp & (-temp)
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                deg = (conflict_mask[v] & rem).bit_count()
                candidates.append(v)
                deg_list.append(deg)
                temp ^= lsb
            
            # Sort by degree
            sorted_pairs = sorted(zip(deg_list, candidates))
            
            # Pick from top 3 (or all if fewer)
            k = min(3, len(sorted_pairs))
            if k == 0:
                break
            chosen = sorted_pairs[random.randrange(k)][1]
            
            sol.append(chosen)
            neighbors = conflict_mask[chosen]
            rem &= ~(1 << (chosen - 1))
            rem &= ~neighbors
        
        # Apply polish to this solution
        sol_mask = 0
        for v in sol:
            sol_mask |= (1 << (v - 1))
        sol_mask, added = polish(sol_mask)
        if added > 0:
            sol = [v for v in vertices if sol_mask & (1 << (v - 1))]
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_sol_set = set(sol)
            best_mask = sol_mask
    
    # Final polish pass on the best solution
    best_mask, added = polish(best_mask)
    if added > 0:
        best_sol = [v for v in vertices if best_mask & (1 << (v - 1))]
        best_size = len(best_sol)
        best_sol_set = set(best_sol)
    
    return best_sol
