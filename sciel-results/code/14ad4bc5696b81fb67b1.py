import random
import time
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    # Build position map
    edge_pos = {}
    for idx, (u, v) in enumerate(P['edge_order'], 1):
        if u > v:
            u, v = v, u
        edge_pos[(u, v)] = idx
    
    def get_pos(u, v):
        if u > v:
            u, v = v, u
        return edge_pos.get((u, v), 1000)

    def calc_cost(perm):
        max_pos = 0
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            p = get_pos(u, v)
            if p > max_pos:
                max_pos = p
        return max_pos

    # 1. Greedy Construction
    # Start with vertex 0
    current_path = [0]
    remaining = set(range(1, n))
    
    # We want to build a path. The cost is the max edge arrival index.
    # Greedy strategy: At each step, add the vertex that minimizes the new max edge cost
    # considering the connection to the end of the path and potentially the start (if we were closing).
    # However, since we are building a cycle, we can just build a Hamiltonian path first,
    # then close it. Or build the cycle directly.
    # Let's build a path. The "cost" of a partial path is the max edge index in it.
    # When adding a vertex v to the end of path P, the new edge is (P[-1], v).
    # New cost = max(old_cost, get_pos(P[-1], v)).
    
    # To get a good cycle, we should also consider the edge back to the start if we were to close now,
    # but that's not the only constraint. A better greedy is to minimize the max of (last-v) and (v-first) ?
    # No, the cycle isn't closed yet.
    # Let's just minimize the max edge index in the path so far.
    
    current_max_cost = 0
    
    for _ in range(n - 1):
        last_v = current_path[-1]
        best_v = None
        best_new_cost = float('inf')
        
        for v in remaining:
            # Edge last_v -> v
            c1 = get_pos(last_v, v)
            # Potential edge v -> first_v if we were to close? 
            # If we just look at path cost, it's max(current_max_cost, c1).
            # But if we are building a cycle, the final edge is important.
            # Let's use a heuristic: minimize max(c1, get_pos(v, current_path[0])) 
            # This encourages closing the cycle with low cost edges too.
            c2 = get_pos(v, current_path[0])
            new_cost = max(current_max_cost, c1, c2)
            
            if new_cost < best_new_cost:
                best_new_cost = new_cost
                best_v = v
        
        if best_v is None:
            # Fallback if something went wrong
            best_v = next(iter(remaining))
            new_cost = max(current_max_cost, get_pos(last_v, best_v), get_pos(best_v, current_path[0]))
        
        current_path.append(best_v)
        remaining.remove(best_v)
        current_max_cost = new_cost

    # Now we have a path. Close it to form a cycle.
    # The cost is the max of the path edges and the closing edge.
    # We can try to reorder the path slightly or just use it as the initial solution for local search.
    
    best_perm = current_path[:]
    best_cost = calc_cost(best_perm)
    
    # 2. Local Refinement (Hill Climbing with 2-opt and Or-opt)
    rng = random.Random(614)
    start_time = time.time()
    time_limit = 50.0
    
    current_perm = best_perm[:]
    current_cost = best_cost
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try 2-opt moves
        for i in range(n):
            for j in range(i + 1, n):
                if j == i: continue
                # Reverse segment i..j
                new_perm = current_perm[:]
                new_perm[i:j+1] = reversed(new_perm[i:j+1])
                new_cost = calc_cost(new_perm)
                if new_cost < current_cost:
                    current_perm = new_perm
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = current_perm[:]
            continue
        
        # Try Or-opt moves (move a segment of length 1, 2, or 3)
        for length in [1, 2, 3]:
            for i in range(n):
                # Extract segment
                segment = []
                for k in range(length):
                    segment.append(current_perm[(i + k) % n])
                
                # Remove segment
                # This is tricky with circular permutation.
                # Let's linearize, remove, reinsert, then re-circularize.
                # Actually, easier: 
                # 1. Identify indices.
                # 2. Create new list without segment.
                # 3. Insert segment at new position.
                
                # To keep it simple and fast, let's just do 2-opt and 3-opt (reversal of 3 nodes is same as 2-opt on adjacent? No.)
                # Or-opt is powerful. Let's implement a simple version.
                
                # If we remove nodes at indices i, i+1, ..., i+length-1
                # We need to handle wrap-around.
                # Let's just try inserting a single node (length=1) for now, as it's common.
                if length > 1: continue # Skip for now to save time, 2-opt is strong.
                
                v = current_perm[i]
                # Remove v
                new_perm = current_perm[:]
                del new_perm[i]
                
                # Insert v at position j
                for j in range(len(new_perm)):
                    # Insert before j
                    candidate = new_perm[:]
                    candidate.insert(j, v)
                    c = calc_cost(candidate)
                    if c < current_cost:
                        current_perm = candidate
                        current_cost = c
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_perm = current_perm[:]
            continue
            
        # If no improvement found in a full pass, we might be stuck in a local optimum.
        # To escape, we could add a random perturbation (Simulated Annealing style), 
        # but the directive says "Hill Climbing". 
        # However, pure hill climbing often gets stuck. 
        # I will add a "random restart" or "kick" if no improvement for a while?
        # No, let's stick to the "approach family". Greedy + Local Search.
        # If stuck, we can just stop or try a different random seed for the greedy part?
        # The greedy part is deterministic given the tie-breaking. 
        # Let's add a small random perturbation if no improvement is found for 1000 iterations?
        # Or just break if no improvement?
        
        # Let's add a counter for no-improvement steps.
        # Actually, the loop above breaks out of inner loops if improved.
        # If not improved, it finishes the loops.
        # We should check if we've been stuck.
        
        # For simplicity and robustness within time limit, if we don't improve in a full sweep, 
        # we can try a random 2-opt move and accept it even if worse (kick), 
        # but that's SA. 
        # Let's just run the hill climbing until no improvement is possible.
        # Given n=24, the space is small enough that hill climbing might find a good local optimum quickly.
        # To avoid infinite loop if no improvement, we break.
        
        break

    return best_perm
# EVOLVE-BLOCK-END
