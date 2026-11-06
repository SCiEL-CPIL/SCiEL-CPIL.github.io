"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin

    N = 256
    # Build conflict graph: edge (a,b) if a+b is a perfect square (including a=b)
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # adjacency as sets for fast lookup
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N and b != a:
                adj[a].add(b)
                adj[b].add(a)
            elif 1 <= b <= N and b == a:
                # self-loop: 2a is a perfect square -> a cannot be in set
                # We'll handle this by marking a as "forbidden"
                pass

    # Numbers with self-loops (2a is a perfect square) are forbidden
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    # Valid vertices
    valid = [x for x in range(1, N + 1) if x not in forbidden]

    # Build bipartite conflict graph: odd vs even
    # Edge between odd a and even b if a+b is a square
    odd_verts = [x for x in valid if x % 2 == 1]
    even_verts = [x for x in valid if x % 2 == 0]

    # Map to indices
    odd_idx = {v: i for i, v in enumerate(odd_verts)}
    even_idx = {v: i for i, v in enumerate(even_verts)}

    # Build bipartite edges: for each odd, find even neighbors
    # edge_list: (odd_index, even_index)
    edge_list = []
    for a in odd_verts:
        for b in adj[a]:
            if b % 2 == 0 and b in even_idx:
                edge_list.append((odd_idx[a], even_idx[b]))

    # Also need: edges between odd-odd or even-even? 
    # Wait: a+b is square. If a is odd, b must be odd for a+b to be even... 
    # Actually odd+odd=even, odd+even=odd, even+even=even.
    # Squares can be odd or even. So we CAN have odd-odd and even-even edges!
    # This means the graph is NOT bipartite in the odd/even sense.
    
    # Let me reconsider. The graph has edges where a+b is a perfect square.
    # This is NOT necessarily bipartite. Let me check:
    # 1+3=4 (odd+odd), 1+8=9 (odd+even), 2+7=9 (even+odd), 2+14=16 (even+even)
    # So it's not bipartite by parity.
    
    # Alternative: use a different bipartition. 
    # Actually, let's just use a general approach.
    # Since the graph is sparse (each vertex has at most ~10 neighbors), 
    # let's use a different strategy.

    # Strategy: Start with a large independent set using greedy, then use 
    # simulated annealing with 1-flip and 2-swap moves.

    # Phase 1: Greedy construction (min-degree first)
    current = set()
    remaining = list(valid)
    # Sort by degree (number of conflicts among valid vertices)
    def degree_in_valid(x, vertex_set):
        return len(adj[x] & vertex_set)
    
    # Use a priority-queue-like approach: repeatedly pick vertex with min conflicts
    # But for speed, let's do a simple greedy: sort by total degree, add if no conflict
    remaining.sort(key=lambda x: len(adj[x] & set(valid)))
    
    for x in remaining:
        if time.time() > deadline:
            break
        if not (adj[x] & current):
            current.add(x)

    best_set = set(current)
    best_size = len(current)

    # Phase 2: Simulated annealing with 1-flip and 2-swap moves
    # State: current independent set
    # Move types:
    #   1-flip: remove one vertex (always valid, decreases size by 1)
    #   2-swap: remove one, add one (net 0)
    #   2-flip-add: remove one, add two (net +1) -- this is the key move
    #   Actually, let's use: 
    #     - Add a vertex (if it doesn't conflict) -- increases size
    #     - Remove a vertex -- decreases size  
    #     - Swap: remove one, add one
    #   The key is to try to ADD vertices. If we can't add any, we need to 
    #   remove some to make room.
    
    # Better approach: 
    #   - Try to add a random vertex from the complement. If it conflicts, 
    #     consider removing the conflicting vertices.
    #   - Use a "repair" heuristic: when we want to add x, remove all neighbors of x 
    #     from the set, then add x. Accept if the net change is positive or with 
    #     some probability.
    
    # Let's use a simpler but effective approach:
    #   1. Try to add vertices greedily (if no conflict)
    #   2. If stuck, try "2-for-1" swaps: remove one vertex v, then try to add 
    #      up to 2 vertices that were previously blocked by v.
    
    # Precompute: for each vertex, its neighbors
    # For the annealing, we need to be efficient.
    
    # Let's use a different approach: local search with specific move types
    # Move 1: Add x (if x not in set and no neighbor in set) -> size +1
    # Move 2: Remove x -> size -1
    # Move 3: Swap: remove x, add y (y not in set, no neighbor of y in set\{x}) -> size 0
    # Move 4: 2-for-1: remove x, add y and z -> size +1
    
    # For efficiency, let's precompute which vertices can be added
    # and which vertices' removal would enable additions.
    
    # Simulated annealing:
    # Temperature starts high, decreases geometrically.
    # At each step:
    #   - Try to add a random vertex from complement. If conflicts, 
    #     compute the set of conflicting vertices. If |conflicts| <= 2, 
    #     remove them and add the new vertex. Accept with probability 
    #     based on net change and temperature.
    
    # For speed, let's limit the number of iterations.
    
    # Actually, let's try a more structured approach:
    # Phase 2a: Iterative improvement with 1-add and 2-for-1 moves
    # Phase 2b: If time permits, simulated annealing for further improvement
    
    # Phase 2a: Iterative improvement
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try 1-add: find a vertex that can be added
        added_any = False
        # Randomize the order to escape symmetry
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        for x in rand_order:
            if time.time() > deadline - 10:
                break
            if x in current:
                continue
            if x in forbidden:
                continue
            if not (adj[x] & current):
                current.add(x)
                improved = True
                added_any = True
                break
        
        if added_any:
            continue
        
        # Try 2-for-1: remove one vertex, add two
        # For each vertex v in current, consider removing it.
        # After removing v, find vertices that can be added.
        # If we can add 2 or more, do it.
        
        # For efficiency, only try a subset of vertices per iteration
        current_list = list(current)
        random.shuffle(current_list)
        
        found_2for1 = False
        for v in current_list:
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove v
            temp_set = current - {v}
            
            # Find vertices that can be added to temp_set
            addable = []
            for x in range(1, N + 1):
                if x in temp_set or x in forbidden:
                    continue
                if not (adj[x] & temp_set):
                    addable.append(x)
            
            if len(addable) >= 2:
                # Add the two best candidates (fewest conflicts with each other)
                # Just take the first two for speed
                y, z = addable[0], addable[1]
                # Verify they don't conflict with each other
                if z not in adj[y]:
                    current = temp_set | {y, z}
                    improved = True
                    found_2for1 = True
                    break
                else:
                    # Try other pairs
                    for i in range(len(addable)):
                        for j in range(i + 1, len(addable)):
                            if addable[j] not in adj[addable[i]]:
                                current = temp_set | {addable[i], addable[j]}
                                improved = True
                                found_2for1 = True
                                break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
        
        if found_2for1:
            continue
        
        # Try 1-for-1 swap with net gain: remove v, add w where w was blocked only by v
        # and w allows adding another vertex
        # This is getting complex. Let's try a simpler 1-swap that enables a 2-add.
        
        # Actually, let's just try: for each v in current, remove v, 
        # then greedily add as many as possible. If total > current size, accept.
        for v in current_list:
            if time.time() > deadline - 10:
                break
            
            temp_set = current - {v}
            # Greedily add
            added = 0
            for x in range(1, N + 1):
                if time.time() > deadline - 10:
                    break
                if x in temp_set or x in forbidden:
                    continue
                if not (adj[x] & temp_set):
                    temp_set.add(x)
                    added += 1
            
            if len(temp_set) > len(current):
                current = temp_set
                improved = True
                break
    
    best_set = set(current)
    best_size = len(current)
    
    # Phase 2b: Simulated annealing for further improvement
    if time.time() < deadline - 5:
        # SA parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        # We'll use a simple SA where:
        # - State is the current set
        # - Move: pick a random vertex x. If x in set, remove it. If not, 
        #   try to add it (remove conflicting vertices if few).
        # - Accept/reject based on energy change (negative of size) and temperature
        
        sa_set = set(current)
        sa_best = set(current)
        sa_best_size = len(current)
        
        # Precompute: for each vertex, list of neighbors (for fast conflict check)
        # adj is already built
        
        while time.time() < deadline - 5:
            # Pick a random move
            if random.random() < 0.3:
                # Try to add a vertex
                x = random.randint(1, N)
                if x in sa_set or x in forbidden:
                    continue
                # Find conflicts
                conflicts = adj[x] & sa_set
                if len(conflicts) == 0:
                    new_set = sa_set | {x}
                    delta = 1  # size increases by 1
                    # Always accept
                    sa_set = new_set
                elif len(conflicts) <= 3:
                    # Remove conflicts, add x
                    new_set = (sa_set - conflicts) | {x}
                    delta = 1 - len(conflicts)
                    # Accept if delta > 0 or with probability exp(delta/T)
                    if delta > 0 or random.random() < math.exp(delta / T):
                        sa_set = new_set
            else:
                # Try to remove a vertex
                if not sa_set:
                    continue
                x = random.choice(list(sa_set))
                new_set = sa_set - {x}
                delta = -1
                # Only accept with some probability (to allow exploration)
                if random.random() < math.exp(delta / T):
                    sa_set = new_set
            
            # Update best
            if len(sa_set) > sa_best_size:
                sa_best_size = len(sa_set)
                sa_best = set(sa_set)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Final: use the best set found
    final_set = best_set
    if len(sa_best) > len(final_set):
        final_set = sa_best
    
    # Verify: ensure it's a valid independent set
    # (In case SA made an invalid move)
    valid_final = set()
    for x in sorted(final_set):
        if not (adj[x] & valid_final):
            valid_final.add(x)
    
    # Ensure nonempty
    if not valid_final:
        # Fallback: take the largest independent set we can find quickly
        for x in range(1, N + 1):
            if x not in forbidden and not (adj[x] & valid_final):
                valid_final.add(x)
    
    result = sorted(valid_final)
    return result if result else [1]
# EVOLVE-BLOCK-END
