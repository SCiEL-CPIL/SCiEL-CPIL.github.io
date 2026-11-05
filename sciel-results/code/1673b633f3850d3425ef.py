import time
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build the conflict graph using bitsets
    # Vertex v (1..256) maps to bit (v-1)
    conflicts_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts_mask[i] |= (1 << (j - 1))
                conflicts_mask[j] |= (1 << (i - 1))
    
    def greedy_randomized(remaining_mask, rng):
        """
        Randomized greedy MIS using bitsets.
        """
        solution = []
        rem = remaining_mask
        while rem:
            # Get list of available vertices
            available = []
            m = rem
            while m:
                bit = (m & -m).bit_length() - 1
                available.append(bit + 1)
                m &= m - 1
            
            if not available:
                break
            
            # Compute degree of each available vertex in the remaining subgraph
            best_deg = N + 1
            candidates = []
            
            for v in available:
                deg = bin(conflicts_mask[v] & rem).count('1')
                if deg < best_deg:
                    best_deg = deg
                    candidates = [v]
                elif deg == best_deg:
                    candidates.append(v)
            
            # Random choice among min-degree candidates
            v = rng.choice(candidates)
            solution.append(v)
            # Remove v and its neighbors from remaining
            rem &= ~(1 << (v - 1))
            rem &= ~conflicts_mask[v]
        
        return solution
    
    # Local search: try to add vertices to the solution
    def try_add_vertices(sol_set, remaining_mask):
        """Try to add vertices from remaining that don't conflict with sol_set."""
        added = []
        rem = remaining_mask
        while rem:
            # Find a vertex in rem that doesn't conflict with sol_set
            found = False
            m = rem
            while m:
                bit = (m & -m).bit_length() - 1
                v = bit + 1
                # Check if v conflicts with any vertex in sol_set
                conflict = False
                for u in sol_set:
                    if conflicts_mask[v] & (1 << (u - 1)):
                        conflict = True
                        break
                if not conflict:
                    sol_set.add(v)
                    added.append(v)
                    rem &= ~(1 << (v - 1))
                    rem &= ~conflicts_mask[v]
                    found = True
                    break
                m &= m - 1
            if not found:
                break
        return added
    
    # 1-swap local search
    def local_search_1swap(sol_list, rng, max_iterations=100):
        """Try to improve solution by swapping vertices."""
        sol_set = set(sol_list)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 2:
                break
            # Try removing each vertex and adding a better one
            for idx in range(len(sol_list)):
                if time.time() > deadline - 2:
                    break
                v = sol_list[idx]
                temp_set = set(sol_set)
                temp_set.remove(v)
                # Find best vertex to add
                best_u = None
                best_score = 0
                for u in range(1, N + 1):
                    if u in temp_set:
                        continue
                    # Check if u conflicts with any in temp_set
                    conflict = False
                    for w in temp_set:
                        if conflicts_mask[u] & (1 << (w - 1)):
                            conflict = True
                            break
                    if not conflict:
                        # Score: prefer vertices with fewer conflicts overall
                        score = N - bin(conflicts_mask[u]).count('1')
                        if score > best_score:
                            best_score = score
                            best_u = u
                if best_u is not None:
                    # Swap v with best_u
                    sol_set.remove(v)
                    sol_set.add(best_u)
                    sol_list[idx] = best_u
                    improved = True
                    break
        return sol_list
    
    # Initial solution
    rng = random.Random(42)
    best_sol = greedy_randomized((1 << N) - 1, rng)
    best_size = len(best_sol)
    best_sol_set = set(best_sol)
    
    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        remaining_mask = (1 << N) - 1
        sol = greedy_randomized(remaining_mask, rng)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_sol_set = set(sol)
        
        # Try local search on the best solution
        if best_size > 0 and time.time() < deadline - 10:
            sol_copy = list(best_sol)
            improved_sol = local_search_1swap(sol_copy, rng, max_iterations=50)
            if len(improved_sol) > best_size:
                best_size = len(improved_sol)
                best_sol = improved_sol
                best_sol_set = set(best_sol)
        
        # Try to add vertices to current best solution
        if time.time() < deadline - 5:
            sol_set = set(best_sol)
            # Find vertices not in sol_set that don't conflict
            added = []
            for u in range(1, N + 1):
                if u in sol_set:
                    continue
                conflict = False
                for w in sol_set:
                    if conflicts_mask[u] & (1 << (w - 1)):
                        conflict = True
                        break
                if not conflict:
                    sol_set.add(u)
                    added.append(u)
            if added:
                best_size = len(sol_set)
                best_sol = list(sol_set)
                best_sol_set = sol_set
    
    return best_sol
