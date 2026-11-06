import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Strategy:
    # 1. Generate a good initial snake using a fast greedy heuristic.
    # 2. Use a Backtracking Search with Forward Checking (Constraint-Based).
    #    - Heuristic: Most Constrained Value (MCV) - choose the next vertex 
    #      that leaves the fewest valid options for the subsequent step.
    #    - This "fail-first" approach prunes the search tree aggressively,
    #      allowing the solver to explore deeper branches before hitting dead ends.
    #    - Use bitsets (integers) for visited set for O(1) membership and fast operations.
    
    sys.setrecursionlimit(5000)

    def greedy_initial():
        """Generate a reasonably long snake quickly using a greedy heuristic."""
        path = [0]
        visited = 1 # bit 0 set
        current = 0
        
        for _ in range(200):
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if (visited >> nb) & 1:
                    continue
                # Check chord condition against path[:-1]
                is_chord = False
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                if not is_chord:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            # Simple greedy: pick first valid
            next_v = valid_nexts[0]
            path.append(next_v)
            visited |= (1 << next_v)
            current = next_v
            
        return path

    best_path = greedy_initial()
    best_len = len(best_path) - 1
    best_path_global = list(best_path)
    best_len_global = best_len

    def search_mcv(path, visited, current, time_limit):
        """
        Constraint-based backtracking with MCV heuristic.
        path: list of vertices
        visited: integer bitset
        current: last vertex in path
        """
        if time.time() > time_limit:
            return path
        
        valid_nexts = []
        
        # 1. Find all valid neighbors of 'current'
        for nb in NEIGHBORS[current]:
            if (visited >> nb) & 1:
                continue
            
            # Check chord condition: nb must not be adjacent to any vertex in path[:-1]
            # Optimization: We can break early.
            is_valid = True
            # Iterate backwards might be slightly faster if chords are more likely with recent vertices?
            # Actually, in a snake, chords are more likely with vertices further back? 
            # No, adjacency is local. But any vertex in the path can form a chord.
            # Let's check all.
            for v in path[:-1]:
                if (nb ^ v).bit_count() == 1:
                    is_valid = False
                    break
            
            if is_valid:
                valid_nexts.append(nb)
        
        if not valid_nexts:
            return path
        
        # 2. Apply MCV Heuristic:
        # Score each candidate nb by the number of valid options it leaves for the NEXT step.
        # We want to pick the one with the FEWEST options (most constrained) to fail fast.
        
        def get_mcv_score(nb):
            count = 0
            # For each neighbor nb2 of nb:
            for nb2 in NEIGHBORS[nb]:
                if (visited >> nb2) & 1:
                    continue
                if nb2 == current: # current is already in path, so it's "visited" in the sense of path history
                    continue
                
                # If we add nb, the path becomes path + [nb].
                # nb2 must not be adjacent to any vertex in path (including the new nb? 
                # Wait, nb2 is adjacent to nb by definition. 
                # The condition is: non-consecutive vertices in the final path differ by >= 2 bits.
                # If we add nb2 after nb, nb2 and nb are consecutive.
                # nb2 must not be adjacent to any vertex in path[:-1] (the vertices before nb).
                # Also nb2 must not be adjacent to nb? No, they are consecutive, so they MUST be adjacent (diff 1 bit).
                # So the check is: nb2 must not be adjacent to any v in path[:-1].
                
                is_valid_next = True
                for v in path[:-1]:
                    if (nb2 ^ v).bit_count() == 1:
                        is_valid_next = False
                        break
                
                if is_valid_next:
                    count += 1
            return count

        # Calculate scores
        scored_nexts = []
        for nb in valid_nexts:
            score = get_mcv_score(nb)
            scored_nexts.append((nb, score))
        
        # Sort by score ascending (fail-first)
        # Add a tiny random perturbation to break ties and allow diversity across restarts?
        # Or just deterministic sort for stability within a restart, and rely on random start/restarts for diversity.
        # Let's use deterministic sort for MCV to be consistent.
        scored_nexts.sort(key=lambda x: x[1])
        
        best_sub_path = path
        max_len = len(path)
        
        for nb, _ in scored_nexts:
            if time.time() > time_limit:
                break
            
            new_path = path + [nb]
            new_visited = visited | (1 << nb)
            
            res = search_mcv(new_path, new_visited, nb, time_limit)
            
            if len(res) > max_len:
                max_len = len(res)
                best_sub_path = res
            
            # Optional: If we found a path longer than current global best, update it immediately?
            # The function returns the best path in this subtree. 
            # We can update global best here to allow early termination if we reach theoretical max?
            # Theoretical max is 511. Unlikely.
            
        return best_sub_path

    # Run multiple restarts with different random seeds and starting points
    # To add diversity, we can shuffle the neighbor order in the initial greedy or the MCV sorting?
    # MCV sorting is deterministic. 
    # We can randomize the starting vertex and the initial greedy choice.
    
    num_restarts = 50 # More restarts, shorter time each?
    # Time allocation: 
    # First few restarts can be longer to find a good baseline?
    # Or just uniform?
    # Let's do uniform time slices, but ensure we have time for at least 5-10 restarts.
    # 115s / 50 = 2.3s per restart. This is very fast for deep search.
    # Maybe fewer restarts with more time each?
    # Let's try 20 restarts with ~5s each.
    num_restarts = 20
    time_per_restart = 110.0 / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        rnd = random.Random(i * 100 + 42)
        
        # Random start
        start_v = rnd.randint(0, SIZE - 1)
        
        # Generate a short greedy path to start the search from?
        # Or start from scratch with just [start_v]?
        # Starting from scratch allows the MCV to guide the entire path.
        # However, MCV is expensive. Starting with a greedy path of length 20-30 reduces the search depth needed.
        # Let's generate a greedy path of length ~30 using random choices.
        
        path = [start_v]
        visited = 1 << start_v
        current = start_v
        
        # Greedy pre-fill
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if (visited >> nb) & 1:
                    continue
                is_chord = False
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                if not is_chord:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            # Random choice for diversity
            next_v = rnd.choice(valid_nexts)
            path.append(next_v)
            visited |= (1 << next_v)
            current = next_v
        
        # Now run MCV search from this state
        current_limit = min(time.time() + time_per_restart, deadline - 1)
        
        found_path = search_mcv(path, visited, current, current_limit)
        
        if len(found_path) - 1 > best_len_global:
            best_len_global = len(found_path) - 1
            best_path_global = found_path
            
            # If we found a very long path, we might want to spend more time on it?
            # For now, just keep going.

    return best_path_global

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
