"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result or empty if time runs out
            # But we expect to finish fast.
            break
            
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency structure
        # C is 0..k-1, I is k..n-1
        # neighbors_in_C[v] for v in I
        neighbors_in_C = {v: set() for v in range(k, n)}
        # Also track edges for G_rem construction
        # We need to know which C-C edges are covered.
        
        # Process edges
        # Edges are sorted.
        # C-C edges: u < v < k
        # C-I edges: u < k <= v
        
        c_c_edges = []
        c_i_edges = []
        
        for u, v in edges:
            if u < k and v < k:
                c_c_edges.append((u, v))
            elif u < k and v >= k:
                neighbors_in_C[v].add(u)
                # Note: edges are generated as (u, v) where u < k, v >= k
                # In instance(), cross edges are (u, v) for v in range(k, n) for u in range(k)
                # So u is always in C, v in I.
            elif u >= k and v >= k:
                # Should not happen for split graphs generated this way
                pass
            else:
                # u >= k, v < k?
                # In instance: edges += [(u, v) for v in range(k, n) for u in range(k) ...]
                # So u is from range(k), v is from range(k, n).
                # So u < k, v >= k.
                # The first part is u in range(k), v in range(u+1, k). So u < v < k.
                # So no I-I edges.
                pass

        # Step 1: Create cliques for each v in I
        cliques = []
        # Track which C-C edges are covered by these cliques
        # A C-C edge (u, w) is covered if there exists v in I such that u in N(v) and w in N(v)
        
        # To efficiently find uncovered C-C edges:
        # For each pair (u, w) in C, check if any v in I has both u and w in its neighbor set.
        
        # Pre-compute for each v in I, the set of pairs (u, w) it covers?
        # That's sum(|N(v)|^2) which might be large.
        # Better: For each pair (u, w), iterate through v?
        # Number of C-C pairs is k(k-1)/2. Max k=30 -> 435 pairs.
        # Number of I vertices is n-k. Max 60.
        # 435 * 60 = 26100 checks. Very fast.
        
        covered_cc = set()
        
        # First, generate the I-cliques
        for v in range(k, n):
            n_v = neighbors_in_C[v]
            if len(n_v) > 0:
                # Clique is {v} union N(v)
                clique = [v] + sorted(list(n_v))
                cliques.append(clique)
                
                # Mark C-C edges covered by this clique
                # Any pair (u, w) in N(v) is covered.
                n_v_list = sorted(list(n_v))
                for i in range(len(n_v_list)):
                    for j in range(i + 1, len(n_v_list)):
                        u = n_v_list[i]
                        w = n_v_list[j]
                        covered_cc.add((u, w))
                        
        # Step 2: Identify remaining C-C edges
        # All C-C edges exist in the graph.
        # Remaining edges are those in c_c_edges but NOT in covered_cc.
        remaining_edges = []
        for u, w in c_c_edges:
            if (u, w) not in covered_cc:
                remaining_edges.append((u, w))
                
        # Step 3: Cover remaining_edges with cliques
        if remaining_edges:
            # Build a graph from remaining_edges to facilitate clique finding
            # Vertices are 0..k-1
            # Adjacency list
            adj = {i: set() for i in range(k)}
            for u, w in remaining_edges:
                adj[u].add(w)
                adj[w].add(u)
                
            rem_set = set(remaining_edges)
            
            while rem_set:
                if time.time() > deadline:
                    break
                    
                # Pick an edge
                u, w = next(iter(rem_set))
                
                # Find maximal clique containing u and w in the graph defined by rem_set
                # Start with {u, w}
                clique = {u, w}
                
                # Candidates: neighbors of u AND neighbors of w in the current graph
                # We need to dynamically update candidates as we add vertices.
                
                # Get initial candidates
                # Neighbors of u in rem_set
                n_u = set()
                n_w = set()
                for a, b in rem_set:
                    if a == u: n_u.add(b)
                    if b == u: n_u.add(a)
                    if a == w: n_w.add(b)
                    if b == w: n_w.add(a)
                    
                candidates = n_u.intersection(n_w)
                
                # Sort candidates by degree in rem_set to prioritize "central" vertices
                # Degree calculation
                deg = {i: 0 for i in range(k)}
                for a, b in rem_set:
                    deg[a] += 1
                    deg[b] += 1
                    
                candidates_list = sorted(list(candidates), key=lambda x: deg[x], reverse=True)
                
                # Greedily add vertices
                for v in candidates_list:
                    if v in clique:
                        continue
                    # Check if v is connected to all in clique
                    connected = True
                    for c in clique:
                        if c == v:
                            continue
                        if (min(c, v), max(c, v)) not in rem_set:
                            connected = False
                            break
                    if connected:
                        clique.add(v)
                        
                # Convert to list
                clique_list = sorted(list(clique))
                cliques.append(clique_list)
                
                # Remove edges covered by this clique
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a = clique_list[i]
                        b = clique_list[j]
                        edge = (min(a, b), max(a, b))
                        if edge in rem_set:
                            rem_set.remove(edge)
                            
        # Convert to list of lists of ints
        # Ensure types are correct
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        results.append(final_cliques)
        
    return results

# EVOLVE-BLOCK-END
