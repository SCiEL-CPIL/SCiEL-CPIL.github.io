import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256
    vertices = list(range(1, N + 1))

    # Precompute conflict masks using Python big ints as bitsets
    # conflict_mask[v] has bit j set if (v*j) % (v+j) == 0 and j != v
    # Bit i (0-indexed) represents vertex (i+1)
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    all_mask = (1 << N) - 1  # bits 0..255 all set

    def bit_count(x):
        return x.bit_count()

    def get_vertex(x):
        """Get the lowest set bit position (0-indexed)."""
        return (x & (-x)).bit_length() - 1

    def greedy_min_degree(available, randomize=False, k=1):
        """
        Greedy independent set using min-degree heuristic.
        available: bitmask of available vertices
        Returns: (solution_list, solution_mask)
        """
        sol = []
        sol_mask = 0
        rem = available
        
        while rem:
            # Find vertex with minimum degree in induced subgraph
            if randomize and k > 1:
                # Find top-k min-degree vertices and pick randomly
                temp = rem
                best_deg = float('inf')
                best_v = -1
                # First pass: find min degree
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < best_deg:
                        best_deg = deg
                    t ^= lsb
                
                # Collect vertices with degree within best_deg + small tolerance
                candidates = []
                t = rem
                while t:
                    lsb = t & (-t)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg <= best_deg + 1:  # allow slight tolerance
                        candidates.append(v)
                    t ^= lsb
                
                if not candidates:
                    # fallback: pick lowest
                    lsb = rem & (-rem)
                    v = lsb.bit_length()
                else:
                    kk = min(k, len(candidates))
                    v = candidates[random.randrange(kk)]
            else:
                # Deterministic min-degree
                min_v = -1
                min_deg = float('inf')
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    deg = bit_count(conflict_mask[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp ^= lsb
                v = min_v
            
            if v < 0:
                break
                
            sol.append(v)
            sol_mask |= (1 << (v - 1))
            
            # Remove v and its neighbors from rem
            rem &= ~(1 << (v - 1))
            rem &= ~conflict_mask[v]
        
        return sol, sol_mask

    # --- Phase 1: Get a good initial solution via multiple greedy runs ---
    best_sol = []
    best_mask = 0
    best_size = 0
    
    # Deterministic greedy
    sol, sol_mask = greedy_min_degree(all_mask, randomize=False)
    if len(sol) > best_size:
        best_size = len(sol)
        best_sol = sol
        best_mask = sol_mask

    # Randomized greedy restarts to find better initial solution
    for it in range(300):
        if time.time() > deadline:
            break
        sol, sol_mask = greedy_min_degree(all_mask, randomize=True, k=5)
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_mask = sol_mask
            # If we've done well, do more
            if best_size >= 130:
                break

    # --- Phase 2: Local improvement - try to add vertices ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find all vertices that can be added
        for v in range(1, N + 1):
            if best_mask & (1 << (v - 1)):
                continue
            # Check if v conflicts with any in best_sol
            if conflict_mask[v] & best_mask:
                continue
            # Can add v
            best_mask |= (1 << (v - 1))
            best_size += 1
            improved = True

    # --- Phase 3: Branch and Bound for exact/maximum independent set ---
    # Use a recursive B&B with strong upper bounds
    # We work on the complement: find minimum vertex cover, or directly maximize IS
    
    # Strategy: Start from current best, try to improve via B&B on the "conflict graph"
    # Actually, let's do a full B&B from scratch with good pruning
    
    # For B&B, we need to work efficiently with bitmasks
    # Upper bound: for a set of remaining vertices, the max IS size is bounded by
    # the number of vertices minus the minimum vertex cover (by Konig's theorem for bipartite,
    # but our graph isn't bipartite). Use a greedy coloring bound instead.
    
    # Let's use a different approach: iterative improvement via 2-swap
    # Try removing 1 vertex and adding 2, or removing 2 and adding 3, etc.
    
    # Fast local search: try 1-for-2 swaps
    def try_improve_1for2(best_mask, best_size):
        """Try to find a vertex v to remove and two vertices a,b to add."""
        # For each vertex in best_sol, check if removing it allows adding 2 new vertices
        # This is expensive; instead, try random 1-for-2
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick a random vertex to remove
            sol_list = [i+1 for i in range(N) if best_mask & (1 << i)]
            if not sol_list:
                break
            v_out = random.choice(sol_list)
            new_mask = best_mask & ~(1 << (v_out - 1))
            
            # Find vertices that can be added given new_mask
            addable = []
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if new_mask & bit:
                    continue
                if conflict_mask[v] & new_mask:
                    continue
                addable.append(v)
            
            if len(addable) >= 2:
                # Try adding 2 random addable vertices
                a, b = random.sample(addable, 2)
                # Check a and b don't conflict
                if conflict_mask[a] & (1 << (b - 1)):
                    continue
                new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1))
                return new_mask2, best_size + 1
        
        return None, best_size

    # Try several rounds of 1-for-2 improvements
    for round_num in range(50):
        if time.time() > deadline:
            break
        new_mask, new_size = try_improve_1for2(best_mask, best_size)
        if new_mask is not None:
            best_mask = new_mask
            best_size = new_size
        else:
            # Also try adding single vertices
            added = False
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1
                added = True
            if not added:
                break

    # --- Phase 4: More aggressive B&B on smaller subproblems ---
    # Identify the "hard" part: vertices that are close to being addable
    # Try a focused B&B on a subset
    
    # Actually, let's do a proper branch-and-bound on the full problem
    # but with a time limit, starting from the current best as a lower bound
    
    # B&B state: (available_mask, current_solution_mask, current_size)
    # Upper bound for available: use a greedy coloring of the conflict graph induced by available
    
    def greedy_coloring_bound(available):
        """
        Compute an upper bound on the size of an independent set in the
        subgraph induced by 'available' using greedy coloring.
        The chromatic number is an upper bound on the clique number of the
        complement, which equals the independence number.
        Actually: alpha(G) <= chi(G) for any graph? No, that's not right.
        alpha(G) * chi(G) >= n, so alpha(G) >= n/chi(G). That's a lower bound.
        
        For an upper bound on alpha, we can use: alpha(G) <= n - tau(G) where tau is
        minimum vertex cover. But computing that is hard.
        
        Better: use the fact that in any graph, alpha(G) <= n - (number of edges in a
        maximum matching)... no.
        
        Let's use a simple bound: the independence number is at most the number of
        vertices minus the size of a minimum vertex cover. A greedy vertex cover
        gives an upper bound on tau, hence a lower bound on alpha... circular.
        
        Simplest valid upper bound: alpha <= n (trivial).
        Better: use the Lovasz bound or just use the size of the largest independent
        set we can greedily find as a heuristic, but we need an upper bound.
        
        For B&B, let's use: the upper bound is the current best size + the number
        of vertices in 'available'. This is weak but valid.
        
        A tighter bound: if we can find a matching of size m in the conflict graph
        induced by 'available', then alpha <= |available| - m (since each matched
        edge can contribute at most 1 to the IS).
        """
        # Simple bound: just return the number of available vertices
        return bit_count(available)
    
    # Let's do a simpler but effective approach: iterative deepening with
    # randomized restarts of the greedy, combined with local search.
    
    # Given time constraints, let's do more randomized greedy with different
    # orderings and local polish.
    
    for it in range(5000):
        if time.time() > deadline:
            break
        
        # Randomized greedy with random vertex ordering
        rem = all_mask
        sol = []
        sol_mask = 0
        
        # Randomize the order by permuting vertices
        verts = list(range(1, N + 1))
        random.shuffle(verts)
        
        for v in verts:
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if not (rem & bit):
                continue
            # Check if v can be added (no conflict with current sol)
            if conflict_mask[v] & sol_mask:
                continue
            sol.append(v)
            sol_mask |= bit
            rem &= ~conflict_mask[v]
            rem &= ~bit
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_mask = sol_mask
            best_sol = sol
            # Polish: try to add more
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if best_mask & bit:
                    continue
                if conflict_mask[v] & best_mask:
                    continue
                best_mask |= bit
                best_size += 1

    # Final polish: greedy add
    for v in range(1, N + 1):
        if time.time() > deadline:
            break
        bit = 1 << (v - 1)
        if best_mask & bit:
            continue
        if conflict_mask[v] & best_mask:
            continue
        best_mask |= bit
        best_size += 1

    # Reconstruct solution list from mask
    final_sol = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    # Ensure nonempty
    if not final_sol:
        final_sol = [1]
    
    return final_sol
