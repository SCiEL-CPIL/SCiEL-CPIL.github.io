import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit

    def generate_graph(n, p, seed):
        rng = Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj, n

    def popcount(x):
        return bin(x).count('1')

    def bron_kerbosch(P, X, adj, n, cliques):
        # P and X are bitsets
        # If P and X are empty, R (implicit) is a maximal clique
        if P == 0 and X == 0:
            cliques.append(P)
            return

        # Pivot selection: choose u in P union X with max degree in P
        U = P | X
        if U:
            # Find pivot with max neighbors in P
            best_pivot = -1
            max_deg = -1
            temp_U = U
            while temp_U:
                lsb = temp_U & -temp_U
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & P)
                if deg > max_deg:
                    max_deg = deg
                    best_pivot = v
                temp_U -= lsb
            
            # Candidates are P \ N(pivot)
            candidates = P & ~(adj[best_pivot])
        else:
            candidates = P

        # Iterate over candidates
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_P, new_X, adj, n, cliques)
            
            # Update P and X
            P = P & ~lsb
            X = X | lsb
            
            temp_cand -= lsb

    def solve_instance(adj, n):
        # 1. Enumerate maximal cliques of size >= 2
        cliques = []
        initial_P = (1 << n) - 1
        initial_X = 0
        bron_kerbosch(initial_P, initial_X, adj, n, cliques)
        
        # Filter cliques of size >= 2
        valid_cliques = [c for c in cliques if popcount(c) >= 2]
        
        if not valid_cliques:
            return []

        # 2. Greedy Hitting Set
        # Map vertex -> list of clique indices
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(valid_cliques):
            temp_c = c
            while temp_c:
                lsb = temp_c & -temp_c
                v = lsb.bit_length() - 1
                vertex_to_cliques[v].append(i)
                temp_c -= lsb

        uncovered = set(range(len(valid_cliques)))
        S = 0  # Bitset for hitting set
        S_list = set()
        
        # Precompute counts for local search later? Or just use S_list.
        # Greedy loop
        while uncovered:
            best_v = -1
            max_covered = -1
            for v in range(n):
                if (S >> v) & 1:
                    continue
                covered = 0
                for i in vertex_to_cliques[v]:
                    if i in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_v = v
            
            if best_v == -1:
                break # Should not happen if valid_cliques is not empty and uncovered is not empty
            
            S |= (1 << best_v)
            S_list.add(best_v)
            for i in vertex_to_cliques[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
        
        # 3. Local Search: Removal
        # Maintain counts of how many vertices in S hit each clique
        counts = [0] * len(valid_cliques)
        for v in S_list:
            for i in vertex_to_cliques[v]:
                counts[i] += 1
        
        improved = True
        while improved:
            improved = False
            # Try removing vertices
            for v in list(S_list):
                # Check if v is redundant
                redundant = True
                for i in vertex_to_cliques[v]:
                    if counts[i] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove v
                    S_list.remove(v)
                    S &= ~(1 << v)
                    for i in vertex_to_cliques[v]:
                        counts[i] -= 1
                    improved = True
                    break # Restart scan after removal

        # 4. Local Search: Swaps
        # Try swapping v in S with u not in S if it reduces size (impossible, size stays same) 
        # but maybe helps future removals? 
        # Actually, standard local search for hitting set:
        # 1. Remove redundant.
        # 2. Try swap: remove v, add u. If valid, do it. (Size unchanged, but structure changes).
        # 3. Remove redundant again.
        
        # Given time constraints, we might skip complex swaps or do a simple one.
        # Let's try a few random swaps if time permits.
        
        if time.time() < deadline - 2.0:
            rng_local = Random(42)
            # Limit iterations
            for _ in range(50):
                if not S_list:
                    break
                v = rng_local.choice(list(S_list))
                # Find candidates u not in S
                candidates_u = [u for u in range(n) if u not in S_list]
                if not candidates_u:
                    break
                rng_local.shuffle(candidates_u)
                swapped = False
                for u in candidates_u[:10]: # Try a few
                    # Check if S - {v} + {u} is valid
                    # This is equivalent to: for all cliques containing v, 
                    # (counts[i] - 1 + (1 if u in clique i else 0)) >= 1
                    # i.e., if u is not in clique i, counts[i] must be > 1.
                    valid_swap = True
                    for i in vertex_to_cliques[v]:
                        u_in_c = (valid_cliques[i] >> u) & 1
                        if not u_in_c and counts[i] == 1:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Perform swap
                        # Remove v
                        S_list.remove(v)
                        S &= ~(1 << v)
                        for i in vertex_to_cliques[v]:
                            counts[i] -= 1
                        
                        # Add u
                        S_list.add(u)
                        S |= (1 << u)
                        for i in vertex_to_cliques[u]:
                            counts[i] += 1
                        
                        swapped = True
                        break
                
                if swapped:
                    # Try removal again
                    improved = True
                    while improved:
                        improved = False
                        for v2 in list(S_list):
                            redundant = True
                            for i in vertex_to_cliques[v2]:
                                if counts[i] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                S_list.remove(v2)
                                S &= ~(1 << v2)
                                for i in vertex_to_cliques[v2]:
                                    counts[i] -= 1
                                improved = True
                                break

        return sorted(S_list)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: Greedy Vertex Cover (which hits all edges, thus all cliques of size>=2? 
            # No, VC hits all edges. Any clique of size>=2 contains an edge. So VC is a valid hitting set for cliques of size>=2.
            # It might not be minimal for the clique hitting set problem, but it's a valid solution.
            adj, n_val = generate_graph(n, p, seed)
            # Greedy VC
            # Active vertices
            active = (1 << n_val) - 1
            cover = 0
            # We need to know which edges are still uncovered.
            # Simpler: Just pick vertices with highest degree in the *remaining* graph?
            # Standard Greedy VC:
            # While edges exist:
            #   Pick v with max degree in current graph
            #   Add to cover
            #   Remove v and incident edges
            
            # To do this efficiently with bitsets:
            # We can just simulate.
            # But for fallback, even simpler: return all vertices? No, too large.
            # Let's do a quick greedy.
            
            # We need to track remaining edges.
            # Let's just use the previous program's fallback logic which was a heuristic.
            # Or better, just run the greedy MIS complement?
            # MIS complement is a Vertex Cover.
            # So if we run greedy MIS, we get a VC.
            
            def greedy_mis(adj, n):
                U = (1 << n) - 1
                I = 0
                while U:
                    best_v = -1
                    max_deg = -1
                    temp_U = U
                    while temp_U:
                        lsb = temp_U & -temp_U
                        v = lsb.bit_length() - 1
                        deg = popcount(adj[v] & U)
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                        temp_U -= lsb
                    I |= (1 << best_v)
                    U &= ~(adj[best_v] | (1 << best_v))
                return I

            mis = greedy_mis(adj, n_val)
            vc = ((1 << n_val) - 1) ^ mis
            cover_list = [i for i in range(n_val) if (vc >> i) & 1]
            out.append(sorted(cover_list))
            continue

        adj, n_val = generate_graph(n, p, seed)
        cover_list = solve_instance(adj, n_val)
        out.append(cover_list)

    return out
