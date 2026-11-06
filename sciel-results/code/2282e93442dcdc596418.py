import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    def is_valid_bitset(adj, n):
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # Common neighbors of i and j
                common = adj[i] & adj[j]
                # Remove i and j from common (not strictly necessary for K4 check if we iterate k > j, l > k)
                # But we need to check if any triangle exists in common.
                # For speed, we can just check if common has any edge.
                # Actually, simpler: iterate k, l
                for k in range(j + 1, n):
                    if not (common >> k) & 1:
                        continue
                    # Check if k is connected to any l > k in common
                    # This is O(n^3) in worst case, but bitsets make it fast if we use bit ops
                    # Let's just do the standard loop for correctness first, optimize if needed
                    for l in range(k + 1, n):
                        if (adj[k] >> l) & 1:
                            return False
        # Check Independent Set of 6
        # This is the bottleneck. 
        # For n <= 35, 35C6 ~ 1.6M. 
        # We need a faster way or a smarter check.
        # We can use a recursive branch and bound for independent set?
        # Or simply: if n is small enough, itertools is okay.
        # For n=30, 30C6 = 593775. ~0.5s in Python.
        # If we call this 1000 times, it's 500s. Too slow.
        
        # Alternative: Check independent set via bitset DP?
        # Or, simpler: just check all 6-sets but break early.
        # Actually, for the local search, we can be smarter.
        # We only need to know if it's valid.
        
        # Let's try to optimize the independent set check.
        # We can precompute something? No, graph changes.
        
        # Let's stick to the loop but make it as fast as possible.
        # If n > 30, this might be too slow for many iterations.
        # We will limit the number of iterations based on n.
        
        for combo in itertools.combinations(range(n), 6):
            is_indep = True
            # Use bitset to check edges within combo
            # For each u in combo, check if any v in combo (v>u) is connected
            for idx, u in enumerate(combo):
                # Check connections to other vertices in combo
                # We can mask adj[u] with the combo mask
                # But building the mask is overhead.
                # Let's just do nested loops
                for v_idx in range(idx + 1, 6):
                    v = combo[v_idx]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False
        return True

    # Start with a random graph or a known good one?
    # Start with n=20 random graph, ensure it's valid, then grow/shrink?
    # Actually, start with a valid graph of moderate size, say n=25.
    # Generate random graph until valid.
    
    current_n = 25
    adj = [0] * current_n
    edges = []
    
    def generate_random_valid(n_target):
        while True:
            adj = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if is_valid_bitset(adj, n_target):
                return adj
    
    adj = generate_random_valid(current_n)
    best_n = current_n
    best_adj = adj[:]
    
    # Local Search
    # We want to maximize n.
    # Strategy: Try to add a vertex. If fail, try to remove one? No, we want max n.
    # If we can't add, maybe we should try a different configuration?
    # The problem is that local search on edge flips for fixed n doesn't change n.
    # We need to search over n.
    
    # Better strategy:
    # 1. Start with small n, valid.
    # 2. Try to increase n by 1.
    #    - Add new vertex.
    #    - Determine edges to new vertex.
    #    - If invalid, try to flip some existing edges to fix it?
    #    This is complex.
    
    # Alternative Strategy:
    # Search for a valid graph of size n_target.
    # Start n_target = 20.
    # Try to find a valid graph.
    # If found, n_target += 1.
    # If not found after T seconds, stop.
    
    # This is a sequential search.
    # We need a good heuristic to find valid graphs for larger n.
    
    # Let's use the "Paley-like" or "Randomized Greedy" construction.
    # For each new vertex, choose neighbors greedily to avoid K4 and Ind6.
    
    def greedy_construct(n_target):
        adj = [0] * n_target
        for v in range(n_target):
            # Choose neighbors among 0..v-1
            # Constraints:
            # 1. No K4: For any pair (i,j) in neighbors, i and j must not be connected to a common neighbor k in neighbors.
            #    Actually, if we add v, and v connects to S, then S must be triangle-free? No.
            #    If S contains a triangle (i,j,k), and v connects to all three, we get K4.
            #    So S must be triangle-free? No, S must not contain a triangle that is also a clique in the existing graph?
            #    Wait, if i,j,k form a triangle in G_{v-1}, and v connects to all, then {v,i,j,k} is K4.
            #    So, for any triangle (i,j,k) in G_{v-1}, v cannot be connected to all three.
            # 2. No Ind6: If there is an independent set of size 5 in G_{v-1} that does NOT include v, 
            #    then v must be connected to at least one vertex in that set?
            #    No, if v is not connected to any vertex in an independent set I of size 5, 
            #    then I U {v} is an independent set of size 6.
            #    So v must be connected to at least one vertex in every independent set of size 5 in G_{v-1}.
            
            # This is a hitting set problem for Ind5 and a packing problem for Triangles.
            
            # Greedy:
            # Start with S = {}
            # We want to choose S such that:
            # - S is not a superset of any triangle in G_{v-1}. (Actually, S must not contain any triangle from G_{v-1})
            # - S hits every independent set of size 5 in G_{v-1}.
            
            # Let's try to construct S.
            # Start with S = all possible neighbors? No, that might create K4.
            # Start with S = empty? That might create Ind6.
            
            # Heuristic:
            # 1. Identify all independent sets of size 5 in G_{v-1}. Let this be H.
            # 2. Identify all triangles in G_{v-1}. Let this be T.
            # 3. We need S such that S intersects every set in H, and S contains no set from T.
            
            # This is still hard.
            
            # Simpler Greedy:
            # Randomly assign edges, then repair?
            
            # Let's try a different approach: 
            # Use the fact that Paley graphs are good.
            # But we want to beat 29.
            
            # Let's try to find a graph of size 30.
            # If we can find one, score > 1.235 (30/24.3? No, score is n/seed_n? 
            # Wait, "search score is value/seed". Seed is 1.0? 
            # "1.0 equals the seed program". 
            # Current score 1.2353 corresponds to n=29? 
            # 29 / 23.47? 
            # If seed is n=23, 29/23 = 1.26.
            # If seed is n=24, 29/24 = 1.20.
            # Let's assume the target is to maximize n.
            
            pass
        
    # Given the complexity of exact constraint satisfaction for n>29, 
    # and the 150s limit, the most robust strategy is to return the best known valid graph.
    # The Paley graph P29 is known to be valid.
    # Can we find P30? 
    # Literature says R(4,6) is between 36 and 42? 
    # Wait, "Exoo's 35-vertex graph is the published record".
    # So n=35 is possible.
    # If we can construct n=35, score will be ~35/24 = 1.45.
    
    # How to construct n=35?
    # We can't just guess.
    # We need a search algorithm.
    
    # Let's use a randomized local search that starts from a random graph of size 35 
    # and tries to satisfy the constraints.
    # This is a SAT-like problem.
    
    # Heuristic: 
    # 1. Start with random graph G on 35 vertices.
    # 2. While invalid:
    #    - Find a violating K4 or Ind6.
    #    - Flip an edge to break it.
    #    - Repeat.
    
    # This is a standard local search for constraint satisfaction.
    
    def find_valid_graph(n_target, time_limit):
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Precompute helpers for speed
        def find_violation(adj, n):
            # Find K4
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1: continue
                    for k in range(j + 1, n):
                        if not (adj[i] >> k) & 1 or not (adj[j] >> k) & 1: continue
                        for l in range(k + 1, n):
                            if (adj[i] >> l) & 1 and (adj[j] >> l) & 1 and (adj[k] >> l) & 1:
                                return ('K4', (i, j, k, l))
            # Find Ind6
            for combo in itertools.combinations(range(n), 6):
                is_indep = True
                for idx, u in enumerate(combo):
                    for v_idx in range(idx + 1, 6):
                        v = combo[v_idx]
                        if (adj[u] >> v) & 1:
                            is_indep = False
                            break
                    if not is_indep: break
                if is_indep:
                    return ('IND6', combo)
            return None
            
        def flip_edge(adj, u, v):
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        current_time = time.time()
        while time.time() - current_time < time_limit:
            viol = find_violation(adj, n_target)
            if viol is None:
                return adj
            # Pick a random edge from the violation to flip
            if viol[0] == 'K4':
                # Pick random pair from the K4 to remove
                i, j, k, l = viol[1]
                pairs = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                u, v = random.choice(pairs)
            else:
                # Pick random pair from Ind6 to add
                combo = viol[1]
                pairs = [(combo[i], combo[j]) for i in range(6) for j in range(i+1, 6)]
                u, v = random.choice(pairs)
            flip_edge(adj, u, v)
            
        return None

    # Try to find a graph of size 35
    # If fails, try 34, 33... down to 29.
    
    for n_target in range(35, 28, -1):
        if time.time() > deadline - 10:
            break
        t_start = time.time()
        # Allow more time for larger n? Or fixed?
        # Let's give 20s for n=35, 10s for 34, etc.
        time_limit = 20.0 if n_target >= 35 else 10.0
        if time.time() + time_limit > deadline:
            time_limit = deadline - time.time() - 5
        if time_limit < 1:
            continue
            
        adj = find_valid_graph(n_target, time_limit)
        if adj:
            # Convert to edges
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
            
    # Fallback to Paley 29
    q = 29
    n = q
    squares = {x * x % q for x in range(1, q)}
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in squares:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
    return {'n': n, 'edges': edges}

