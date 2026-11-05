import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        # Bitmask representation for adjacency
        # adj_mask[i] is an integer where bit j is 1 if (i,j) is an edge
        adj_mask = [0] * n
        for u, v in edges_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Helper to check if a set of vertices (bitmask) is a clique
        def is_clique(mask):
            # For each vertex in mask, check if it is connected to all others
            # This is O(n^2) in worst case, but masks are fast
            temp = mask
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Check if v is connected to all other vertices in mask
                # mask without v
                rest = mask ^ lsb
                # adj_mask[v] & rest should equal rest
                if (adj_mask[v] & rest) != rest:
                    return False
            return True

        # Helper to get list of vertices from mask
        def mask_to_list(mask):
            return [i for i in range(n) if (mask >> i) & 1]

        # Greedy Max Clique Solver (Bitmask version) to get initial upper bound
        def greedy_max_clique(mask):
            # Returns a bitmask of a maximal clique
            if mask == 0:
                return 0
            
            # Simple heuristic: pick vertex with max degree in mask
            best_v = -1
            best_deg = -1
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = bin(adj_mask[v] & mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return 0
                
            # Recursively find max clique in neighbors of best_v
            # Actually, for greedy, we just build a clique greedily
            clique_mask = 1 << best_v
            candidates = adj_mask[best_v] & mask
            
            # Greedy add vertices
            # We want to add vertices that are connected to all in clique_mask
            # and have high degree in remaining candidates
            while candidates:
                # Pick vertex with max degree in candidates
                # And connected to all in clique_mask
                best_next = -1
                best_next_deg = -1
                temp_cand = candidates
                while temp_cand:
                    lsb = temp_cand & -temp_cand
                    v = lsb.bit_length() - 1
                    temp_cand ^= lsb
                    # Check if v connected to all in clique_mask
                    if (adj_mask[v] & clique_mask) == clique_mask:
                        deg = bin(adj_mask[v] & candidates).count('1')
                        if deg > best_next_deg:
                            best_next_deg = deg
                            best_next = v
                            
                if best_next == -1:
                    break
                
                clique_mask |= (1 << best_next)
                # Update candidates: must be connected to new vertex
                candidates &= adj_mask[best_next]
                
            return clique_mask

        # Get initial greedy solution
        remaining_mask = 0
        for i in range(n):
            # We need to represent edges as a mask? 
            # It's easier to keep the set of edges for the partition logic, 
            # but use bitmasks for clique checks.
            pass
        
        # Let's use a set of edges for the main loop state, 
        # but bitmasks for clique validation and generation.
        
        # Greedy Phase
        def run_greedy():
            rem_edges = set(edges_set)
            cliques = []
            # Build adj list for current rem_edges
            # To speed up, we can just use the global adj_mask and check if edge is in rem_edges
            
            while rem_edges:
                if time.time() > DEADLINE:
                    break
                # Find a max clique in the graph induced by rem_edges
                # Vertices involved in rem_edges
                active_verts = set()
                for u, v in rem_edges:
                    active_verts.add(u)
                    active_verts.add(v)
                
                if not active_verts:
                    break
                    
                # Get bitmask of active vertices
                active_mask = 0
                for v in active_verts:
                    active_mask |= (1 << v)
                
                # Find max clique in subgraph induced by active_mask, 
                # but restricted to edges in rem_edges.
                # Note: adj_mask contains ALL edges. 
                # We need a clique where ALL edges are in rem_edges.
                # This is tricky. A clique in G might have edges not in rem_edges.
                # So we must ensure the clique is a clique in the *remaining* graph.
                
                # Let's construct the remaining adjacency mask
                rem_adj_mask = [0] * n
                for u, v in rem_edges:
                    rem_adj_mask[u] |= (1 << v)
                    rem_adj_mask[v] |= (1 << u)
                    
                # Find max clique using rem_adj_mask
                # We can use a simple greedy on rem_adj_mask
                best_clique_mask = 0
                # Try each vertex as a seed
                # To save time, just pick one with max degree
                max_deg_v = -1
                max_deg = -1
                for v in active_verts:
                    deg = bin(rem_adj_mask[v]).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        max_deg_v = v
                        
                if max_deg_v != -1:
                    # Greedy expansion from max_deg_v
                    c_mask = 1 << max_deg_v
                    cands = rem_adj_mask[max_deg_v]
                    while cands:
                        # Pick vertex in cands with max degree in cands
                        # and connected to all in c_mask
                        best_next = -1
                        best_next_deg = -1
                        temp_c = cands
                        while temp_c:
                            lsb = temp_c & -temp_c
                            v = lsb.bit_length() - 1
                            temp_c ^= lsb
                            if (rem_adj_mask[v] & c_mask) == c_mask:
                                deg = bin(rem_adj_mask[v] & cands).count('1')
                                if deg > best_next_deg:
                                    best_next_deg = deg
                                    best_next = v
                        if best_next == -1:
                            break
                        c_mask |= (1 << best_next)
                        cands &= rem_adj_mask[best_next]
                    best_clique_mask = c_mask
                else:
                    # Fallback: pick any edge
                    u, v = next(iter(rem_edges))
                    best_clique_mask = (1 << u) | (1 << v)
                
                clique_verts = mask_to_list(best_clique_mask)
                if len(clique_verts) < 2:
                    # Should not happen if rem_edges not empty
                    u, v = next(iter(rem_edges))
                    clique_verts = [u, v]
                    best_clique_mask = (1 << u) | (1 << v)
                
                cliques.append(clique_verts)
                
                # Remove edges of clique from rem_edges
                # Also update rem_adj_mask? 
                # For simplicity, we just rebuild rem_adj_mask in the next iteration 
                # or remove edges from rem_edges and rely on that.
                # Rebuilding is O(E). E is small.
                for i in range(len(clique_verts)):
                    for j in range(i + 1, len(clique_verts)):
                        u, v = clique_verts[i], clique_verts[j]
                        if u > v:
                            u, v = v, u
                        rem_edges.discard((u, v))
                        
            return cliques

        greedy_sol = run_greedy()
        if not greedy_sol:
            return []
            
        best_count = len(greedy_sol)
        best_sol = greedy_sol
        
        # B&B Phase: Try to find a solution with fewer cliques
        # Limit search to best_count - 1, then -2, etc.
        # We use a global best found in B&B
        
        # State for B&B: 
        # rem_edges: set of edges
        # cliques: list of cliques chosen so far
        # limit: max allowed cliques
        
        # To make B&B efficient, we need good branching.
        # Branch on an edge (u,v). 
        # Candidates: All maximal cliques containing (u,v) in the remaining graph.
        
        def get_maximal_cliques_containing_edge(u, v, rem_edges, rem_adj_mask):
            # Returns a list of bitmasks of maximal cliques containing u and v
            # in the graph defined by rem_adj_mask
            # Start with {u, v}
            base_mask = (1 << u) | (1 << v)
            
            # Candidates are common neighbors of u and v in rem_adj_mask
            cands = rem_adj_mask[u] & rem_adj_mask[v]
            
            cliques = []
            
            # Recursive expansion
            def expand(current_mask, candidates):
                # Check if current_mask is maximal
                # If no candidate can be added, it's maximal
                # Actually, we want to generate ALL maximal cliques? 
                # That's expensive. 
                # Heuristic: Generate a few good ones.
                # For B&B, we need to explore enough.
                # Let's just return the current mask if we can't add more?
                # No, we should explore adding vertices.
                
                # If time is up, return current
                if time.time() > DEADLINE:
                    cliques.append(current_mask)
                    return
                    
                # Try to add vertices from candidates
                # To avoid duplicates, we can enforce an ordering?
                # For now, let's just pick the best candidate and recurse, 
                # and also try stopping here (current mask is a clique).
                
                # Option 1: Stop here (current_mask is a valid clique)
                # Check if it's maximal? 
                # If not maximal, we might miss a better partition?
                # Actually, using a non-maximal clique is allowed, but usually suboptimal.
                # However, sometimes a smaller clique allows a better global structure.
                # So we should consider current_mask.
                cliques.append(current_mask)
                
                # Option 2: Add a vertex
                # Pick a candidate
                if not candidates:
                    return
                
                # Pick candidate with max degree in candidates
                best_next = -1
                best_next_deg = -1
                temp_c = candidates
                while temp_c:
                    lsb = temp_c & -temp_c
                    w = lsb.bit_length() - 1
                    temp_c ^= lsb
                    if (rem_adj_mask[w] & current_mask) == current_mask:
                         deg = bin(rem_adj_mask[w] & candidates).count('1')
                         if deg > best_next_deg:
                             best_next_deg = deg
                             best_next = w
                             
                if best_next != -1:
                    new_mask = current_mask | (1 << best_next)
                    new_cands = candidates & rem_adj_mask[best_next]
                    # Remove best_next from candidates to avoid cycles/duplicates in this branch
                    # Actually, we are doing a DFS.
                    expand(new_mask, new_cands)
                    
            expand(base_mask, cands)
            return cliques

        # B&B Search
        def search_bnb(rem_edges, cliques, limit):
            nonlocal best_count, best_sol
            
            if time.time() > DEADLINE:
                return False
                
            if not rem_edges:
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_sol = [list(c) for c in cliques]
                return True # Found a valid solution within limit
                
            if len(cliques) >= limit:
                return False
                
            # Lower bound: 
            # Max edges in a clique of size k is k*(k-1)/2.
            # Max clique size in remaining graph is at most max_degree + 1?
            # Or we can estimate.
            # Simple LB: len(rem_edges) / max_possible_edges_per_clique
            # Max possible edges per clique is bounded by the number of edges in the largest clique.
            # We can just use 1 as a trivial LB (each clique covers at least 1 edge).
            # Better: len(rem_edges) / (max_degree_in_rem + 1) * 2? 
            # Let's use a simple LB: 1 if rem_edges else 0.
            
            if len(cliques) + 1 > limit:
                 return False
                 
            # Pick an edge to branch on
            # Heuristic: Edge with highest "difficulty"
            # e.g., sum of degrees of its endpoints in rem_edges
            u, v = next(iter(rem_edges))
            
            # Get candidates
            # We need rem_adj_mask for the current rem_edges
            rem_adj_mask = [0] * n
            for a, b in rem_edges:
                rem_adj_mask[a] |= (1 << b)
                rem_adj_mask[b] |= (1 << a)
                
            candidates = get_maximal_cliques_containing_edge(u, v, rem_edges, rem_adj_mask)
            
            # Sort candidates by size descending (greedy preference)
            candidates.sort(key=lambda m: bin(m).count('1'), reverse=True)
            
            for c_mask in candidates:
                c_verts = mask_to_list(c_mask)
                if len(c_verts) < 2:
                    continue
                    
                # Check if c_mask is a valid clique in rem_edges
                # (It should be by construction, but double check)
                valid = True
                for i in range(len(c_verts)):
                    for j in range(i + 1, len(c_verts)):
                        a, b = c_verts[i], c_verts[j]
                        if a > b: a, b = b, a
                        if (a, b) not in rem_edges:
                            valid = False
                            break
                    if not valid: break
                if not valid:
                    continue
                
                # Remove edges of c_mask from rem_edges
                new_rem_edges = set(rem_edges)
                for i in range(len(c_verts)):
                    for j in range(i + 1, len(c_verts)):
                        a, b = c_verts[i], c_verts[j]
                        if a > b: a, b = b, a
                        new_rem_edges.discard((a, b))
                        
                # Recurse
                if search_bnb(new_rem_edges, cliques + [c_verts], limit):
                    return True
                    
            return False

        # Try to improve
        # Start with limit = best_count - 1
        # If found, try best_count - 2, etc.
        # To save time, we only do a few iterations or one deep search.
        
        # Let's try one B&B run with limit = best_count - 1
        # If it finds a solution, we update best.
        # We might not have time for multiple.
        
        # To increase chance of success, we can randomize the edge selection in B&B?
        # Or just run it once.
        
        # If greedy was very good (e.g. 5 cliques), B&B with limit 4 might be hard.
        # But for dense graphs, the gap is often small.
        
        # Let's run B&B
        # We need to be careful with recursion depth and time.
        
        # To make it more robust, we can add a time check in the loop.
        
        # Let's just run the B&B once.
        # If it times out, we return the greedy solution.
        
        # Note: The B&B above might be slow if the branching factor is high.
        # For dense graphs, the number of maximal cliques containing an edge might be small?
        # Actually, in a dense graph, there are many cliques.
        # But we are looking for a *partition*.
        
        # Alternative: Just return the greedy solution if B&B is too risky.
        # Given the "Stagnation" warning, a structural change is needed.
        # B&B is a structural change.
        
        # Let's execute B&B with a tight time limit.
        bnb_start = time.time()
        BNB_TIME_LIMIT = 10.0 # Spend max 10s on B&B
        
        # We need to modify search_bnb to respect a local deadline
        # The global DEADLINE is 115.0.
        
        # Let's just call it. If it returns True, we have a better solution.
        # If it returns False, we keep greedy.
        
        # To ensure we don't hang, we rely on the time check inside search_bnb.
        
        # One issue: search_bnb might not find a solution if the limit is too tight.
        # We try limit = best_count - 1.
        
        if best_count > 1:
             # Try to find a solution with best_count - 1 cliques
             # We pass a copy of the state? No, we pass the initial state.
             # rem_edges is edges_set
             # cliques is []
             
             # We need to be careful that search_bnb doesn't modify global state incorrectly.
             # It uses local variables.
             
             # Let's run it.
             # We might want to randomize the edge pick in B&B to escape local minima?
             # No, B&B is exact (if it finishes).
             
             # Let's just run it.
             found = search_bnb(set(edges_set), [], best_count - 1)
             
             # If found, best_sol is updated.
             # We can try to go lower?
             # If time permits.
             if time.time() < DEADLINE - 20.0:
                 if best_count > 2:
                     found2 = search_bnb(set(edges_set), [], best_count - 2)
                     
        return best_sol

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
