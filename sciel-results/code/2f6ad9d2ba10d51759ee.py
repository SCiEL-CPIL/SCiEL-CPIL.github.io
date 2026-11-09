import numpy as np

def furstenberg_sarkozy_205() -> np.ndarray:
        m = 205
        k = 2

        # Compute forbidden differences
        forbidden = set()
        for x in range(1, m):
            forbidden.add(pow(x, k, m))
        forbidden.discard(0)

        # Build adjacency bitsets
        adj = [0] * m
        for i in range(m):
            for f in forbidden:
                j1 = (i + f) % m
                j2 = (i - f) % m
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        for i in range(m):
            adj[i] &= ~(1 << i)

        full_mask = (1 << m) - 1

        # Helper: bit count
        def popcount(x):
            return x.bit_count()

        # Greedy Coloring Bound for MIS
        # We want to color the complement graph \bar{G}.
        # A color class in \bar{G} is a clique in G.
        # We greedily find maximal cliques in G[candidates].
        def greedy_color_bound(candidates):
            if candidates == 0:
                return 0
            
            remaining = candidates
            color_count = 0
            
            while remaining:
                # Find a maximal clique in G induced by 'remaining'
                # Greedy: pick vertex with highest degree in 'remaining'?
                # Or just pick the first one?
                # Standard greedy coloring for upper bound:
                # Order vertices, assign smallest available color.
                # Here we are finding a sequence of maximal cliques that cover the set.
                # This is equivalent to coloring the complement.
                
                clique = 0
                temp = remaining
                # To get a tighter bound, we should try to make the clique as large as possible.
                # Heuristic: pick vertex with max degree in 'temp' to add to clique?
                # If we pick a high-degree vertex, it restricts 'temp' more, potentially leading to smaller subsequent cliques?
                # Actually, for coloring, we want *few* colors. So we want *large* color classes (cliques in G).
                # To make a large clique, we should pick vertices that are connected to many others in the current candidate pool.
                
                while temp:
                    # Find vertex in temp with max degree in temp
                    max_deg = -1
                    best_v = -1
                    t = temp
                    while t:
                        lb = t & (-t)
                        v = lb.bit_length() - 1
                        deg = popcount(adj[v] & temp)
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        t ^= lb
                    
                    if best_v == -1:
                        break
                    
                    lb = 1 << best_v
                    # Check if best_v can be added to clique
                    # It must be adjacent to all in clique
                    if (adj[best_v] & clique) == clique:
                        clique |= lb
                        # Restrict temp to neighbors of best_v
                        temp &= adj[best_v]
                    
                    # Remove best_v from temp
                    temp &= ~lb

                color_count += 1
                remaining &= ~clique
                
            return color_count

        best_size = 0
        best_set = 0

        def branch_and_bound(candidates, current_set, current_size):
            nonlocal best_size, best_set
            
            if candidates == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
            
            ub = current_size + greedy_color_bound(candidates)
            if ub <= best_size:
                return
            
            # Branching heuristic: pick vertex with max degree in candidates
            max_deg = -1
            best_v = -1
            t = candidates
            while t:
                lb = t & (-t)
                v = lb.bit_length() - 1
                deg = popcount(adj[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                t ^= lb
            
            if best_v == -1:
                return
                
            # Branch 1: Include best_v
            new_candidates = candidates & ~(1 << best_v) & ~adj[best_v]
            branch_and_bound(new_candidates, current_set | (1 << best_v), current_size + 1)
            
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            branch_and_bound(new_candidates, current_set, current_size)

        # Initial Greedy Solution
        # Sort by degree ascending? Usually for Independent Set, picking low-degree vertices first is safer?
        # Or high degree? 
        # Let's try sorting by degree descending and picking if possible?
        # Actually, a common heuristic is "Greedy by degree".
        # Let's just use the simple loop but maybe sorted by degree.
        vertices = list(range(m))
        # Sort by degree in full graph
        vertices.sort(key=lambda x: popcount(adj[x]), reverse=True)
        
        greedy_set = 0
        for v in vertices:
            if (adj[v] & greedy_set) == 0:
                greedy_set |= (1 << v)
        
        best_size = popcount(greedy_set)
        best_set = greedy_set

        branch_and_bound(full_mask, 0, 0)

        result = []
        for v in range(m):
            if (best_set >> v) & 1:
                result.append(v)

        return np.array(result, dtype=np.int64)
