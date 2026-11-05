"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140 

    def random_graph(n, p, seed):
        """Generate a random graph and return adjacency as a list of bitmasks."""
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bits_to_list(mask):
        """Helper to get list of vertices from a bitmask."""
        verts = []
        m = mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        return verts

    def solve_instance(n, p, seed, time_limit):
        adj = random_graph(n, p, seed)
        rem = adj[:]  # Remaining edges as bitmasks
        cliques = []
        
        # Check if graph is empty
        if all(x == 0 for x in rem):
            return cliques

        # Precompute initial spectral order
        def compute_spectral_order(adj_list):
            if n == 0:
                return []
            A = np.zeros((n, n), dtype=np.float64)
            for u in range(n):
                mask = adj_list[u]
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    A[u, v] = 1.0
                    m ^= lsb
            
            vec = np.ones(n, dtype=np.float64) / n
            for _ in range(50):
                new_vec = A @ vec
                norm = np.linalg.norm(new_vec)
                if norm == 0:
                    break
                new_vec /= norm
                if np.linalg.norm(new_vec - vec) < 1e-10:
                    break
                vec = new_vec
            
            order = np.argsort(-vec)
            return [int(i) for i in order]

        spectral_order = compute_spectral_order(adj)
        spectral_rank = [0] * n
        for rank, v in enumerate(spectral_order):
            spectral_rank[v] = rank

        # --- Core Search: Bron-Kerbosch with Coloring ---
        def find_max_clique():
            """
            Find a maximum clique in the graph defined by 'rem' bitmasks.
            Uses Bron-Kerbosch with coloring.
            """
            # Active vertices: those with at least one remaining edge
            active_mask = 0
            for i in range(n):
                if rem[i]:
                    active_mask |= (1 << i)

            if active_mask == 0:
                return []

            best_clique = []
            
            def color_sort(P):
                """
                Sort vertices in P by color for pruning.
                Returns a list of (color, vertex) sorted by color ascending.
                """
                verts = bits_to_list(P)
                # Sort by degree in P descending, then by spectral rank
                verts.sort(key=lambda v: (-(rem[v] & P).bit_count(), spectral_rank[v]))
                
                colors = {}
                colored = []
                for v in verts:
                    used = set()
                    nm = rem[v] & P
                    while nm:
                        lsb = nm & -nm
                        u = lsb.bit_length() - 1
                        if u in colors:
                            used.add(colors[u])
                        nm ^= lsb
                    c = 1
                    while c in used:
                        c += 1
                    colors[v] = c
                    colored.append((c, v))
                
                colored.sort(key=lambda x: x[0])
                return colored

            def bk(R, P, X):
                nonlocal best_clique
                if time.time() > time_limit:
                    return
                
                if P == 0:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                
                colored = color_sort(P)
                
                # Process in reverse order (highest color first)
                for color, v in reversed(colored):
                    if time.time() > time_limit:
                        return
                    if len(R) + color <= len(best_clique):
                        break
                    
                    R.append(v)
                    bk(R, P & rem[v], X & rem[v])
                    R.pop()
                    
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], active_mask, 0)
            return best_clique

        # --- Set Cover / Greedy Heuristic ---
        def find_greedy_clique():
            """
            Find a good clique by greedily picking vertices that cover the most 
            currently uncovered edges. This acts as a fast heuristic or alternative 
            to exact max clique search, or as a seed.
            """
            # Calculate current uncovered edges count for each vertex
            # Actually, for set cover, we want a clique that covers many edges.
            # A clique of size k covers k*(k-1)/2 edges.
            # So maximizing clique size is equivalent to maximizing covered edges 
            # IF we assume we only pick cliques.
            # However, the "Set Cover" framing suggests we might want to prioritize 
            # cliques that cover edges that are "hard" to cover or have high weight.
            # Here, all edges have weight 1. So Max Clique is the optimal greedy choice 
            # for a single step.
            
            # But maybe we can use a heuristic to find a *large* clique faster than 
            # exact BK if BK is too slow? Or use it to break ties?
            
            # Let's try a simple greedy expansion:
            # 1. Pick vertex with highest degree in rem.
            # 2. Intersect neighbors.
            # 3. Repeat.
            
            if not any(rem):
                return []
                
            # Start with vertex having highest degree
            start_v = max(range(n), key=lambda v: rem[v].bit_count())
            if rem[start_v].bit_count() == 0:
                return []
                
            current_clique = [start_v]
            candidates = rem[start_v]
            
            while candidates:
                if time.time() > time_limit:
                    break
                # Pick candidate with highest degree within candidates
                # We need to check which candidates are connected to each other?
                # No, 'candidates' is the intersection of neighborhoods, so they are 
                # all connected to all previous clique members.
                # We just need to pick one that connects to the most other candidates.
                
                best_v = -1
                best_deg = -1
                m = candidates
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Degree of v within candidates
                    deg = (rem[v] & candidates).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m ^= lsb
                
                if best_v == -1:
                    break
                    
                current_clique.append(best_v)
                # Update candidates: must be neighbors of best_v AND in current candidates
                candidates &= rem[best_v]
                
            return current_clique

        # Main loop
        # Strategy: Alternate or combine?
        # The prompt suggests: "Model the edge partitioning as a weighted set cover... 
        # solve it using a greedy algorithm that selects the clique covering the 
        # maximum number of currently uncovered edges".
        # Since every clique of size k covers k(k-1)/2 edges, and we want to minimize 
        # number of cliques, picking the largest clique is the correct greedy move 
        # for this specific objective (min count).
        # So find_max_clique is the core.
        
        # However, we can use the "greedy" idea to perhaps speed up or guide.
        # Let's stick to the robust BK for correctness and quality, but ensure 
        # we have a fast fallback if BK is too slow on large instances.
        
        # We will use BK. If it times out, we return what it found (which might be 
        # a good sub-clique) or we fall back to greedy.
        # Actually, BK with coloring is very fast for n=64.
        
        while time.time() < time_limit:
            # Check if edges remain
            has_edges = False
            for i in range(n):
                if rem[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Try to find max clique
            clique = find_max_clique()
            
            # If BK didn't find anything (or timed out without finding a clique?), 
            # use greedy as backup.
            # Note: find_max_clique returns [] if no edges or if it times out before 
            # finding a valid clique larger than 0? 
            # In BK, if P becomes 0, it checks R. If R is empty, best_clique remains [].
            # If it times out, it returns best_clique found so far.
            
            if not clique or len(clique) < 2:
                # Fallback to greedy expansion
                clique = find_greedy_clique()
                
            if not clique or len(clique) < 2:
                # If even greedy fails, break to final fallback
                break

            cliques.append(clique)

            # Remove edges of this clique from rem
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
            
            # Update spectral order periodically?
            # For n=32/64, it's cheap. Let's update every 5 iterations or if graph changes significantly.
            # Simple heuristic: just use static for now, or update if remaining edges < 50% initial.
            # Let's update if we've removed more than half the edges.
            # This requires tracking initial edges.
            # initial_edges = sum(adj[i].bit_count() for i in range(n)) // 2
            # current_edges = sum(rem[i].bit_count() for i in range(n)) // 2
            # if current_edges < 0.5 * initial_edges:
            #     spectral_order = compute_spectral_order(rem)
            #     ...
            # Given n is small, let's just recompute every 10 iterations to be safe? 
            # Or just stick to static. Static worked well in seed.
            # Let's try updating if time permits and graph is sparse enough.
            # Actually, let's keep it simple: static spectral order.

        # Fallback: cover remaining edges as 2-cliques
        for u in range(n):
            mask = rem[u]
            # Only consider neighbors v > u to avoid duplicates
            mask &= ~((1 << (u + 1)) - 1) 
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v and rem[u] & (1 << v):
                    cliques.append([u, v])
                    # Remove this edge
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
                m ^= lsb

        return cliques

    # Solve first instance (n=32)
    # Budget allocation: give 60% of time to first, 40% to second?
    # Or just run until time limit for each, but total must be < 140s.
    # Let's give 70s to first, 70s to second.
    
    budget1 = 70
    cliques1 = solve_instance(32, 0.6, 601, start_time + budget1)
    
    # Update deadline for second instance
    current_time = time.time()
    remaining_budget = deadline - current_time
    if remaining_budget < 10:
        # Not enough time, just return what we have for second as empty or minimal
        # But we must return valid partitions. 
        # If we don't have time, we should solve it very fast or fallback.
        # Let's give it whatever is left.
        budget2 = max(1, remaining_budget - 2)
    else:
        budget2 = 70

    cliques2 = solve_instance(64, 0.55, 602, start_time + budget1 + budget2)

    return [cliques1, cliques2]
# EVOLVE-BLOCK-END
