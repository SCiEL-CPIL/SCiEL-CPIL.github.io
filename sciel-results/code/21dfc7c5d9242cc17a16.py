import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        # adj[i] is an integer bitmask of neighbors of i
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = []
        
        # Helper to get degree
        def degree(u):
            return bin(adj[u]).count('1')
        
        # Helper to remove edges of a clique
        def remove_clique_edges(clique):
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        # Main loop
        while True:
            # Check time
            if time.time() > deadline:
                break
                
            # Find vertex with max degree
            max_deg = -1
            u = -1
            for i in range(n):
                d = degree(i)
                if d > max_deg:
                    max_deg = d
                    u = i
            
            if max_deg == 0:
                break # No edges left

            # Find a clique containing u
            clique = [u]
            # Candidates must be neighbors of all vertices in clique
            # Start with neighbors of u
            P = adj[u]
            
            while P:
                # Find v in P with max "local degree" (degree within P)
                # local_deg(v) = popcount(adj[v] & P)
                best_v = -1
                best_score = -1
                # Iterate through bits in P
                pp = P
                while pp:
                    # Get lowest set bit
                    lsb = pp & -pp
                    v = lsb.bit_length() - 1
                    # Calculate score: how many neighbors v has in P
                    score = bin(adj[v] & P).count('1')
                    if score > best_score:
                        best_score = score
                        best_v = v
                    pp &= pp - 1 # Remove lowest set bit
                
                if best_v == -1:
                    break
                
                clique.append(best_v)
                # Update P: must be neighbors of best_v too
                P &= adj[best_v]
                P &= ~(1 << best_v) # Remove best_v from candidates (already done by intersection if self-loop not set, but safe)

            # Add clique to result
            cliques.append(clique)
            # Remove edges
            remove_clique_edges(clique)

        return cliques

    # Solve the two instances
    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
