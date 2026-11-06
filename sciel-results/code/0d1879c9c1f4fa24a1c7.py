"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def is_clique(vertices, adj):
        """Check if a set of vertices forms a clique in the given graph."""
        vs = list(vertices)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[vs[i]]:
                    return False
        return True

    def greedy_partition(adj, n):
        """
        Fast initial partition using a greedy 'largest clique first' approach.
        This is used as the starting point for local search.
        """
        remaining_adj = [set(s) for s in adj]
        cliques = []
        
        # Pre-compute degrees to order vertices
        # We will repeatedly find a maximal clique in the remaining graph
        
        while True:
            if time.time() > deadline - 5.0: # Leave time for verification
                break
            
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if remaining_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find a maximal clique greedily
            # Start with the vertex of highest degree in remaining graph
            best_start = -1
            max_deg = -1
            for u in range(n):
                d = len(remaining_adj[u])
                if d > max_deg:
                    max_deg = d
                    best_start = u
            
            if best_start == -1:
                break
                
            clique = [best_start]
            candidates = set(remaining_adj[best_start])
            
            # Greedily add vertices
            while candidates:
                # Pick candidate with highest degree within candidates
                best_v = -1
                max_cand_deg = -1
                for v in candidates:
                    # Degree within candidates
                    d = len(remaining_adj[v] & candidates)
                    if d > max_cand_deg:
                        max_cand_deg = d
                        best_v = v
                
                if best_v == -1:
                    break
                
                clique.append(best_v)
                # New candidates must be neighbors of best_v
                candidates = candidates & remaining_adj[best_v]
                
            # Add the clique
            cliques.append(frozenset(clique))
            
            # Remove edges covered by this clique from remaining_adj
            for u in clique:
                for v in clique:
                    if u != v:
                        if v in remaining_adj[u]:
                            remaining_adj[u].remove(v)
                        if u in remaining_adj[v]:
                            remaining_adj[v].remove(u)
                        
        # Handle any remaining edges as 2-cliques
        for u in range(n):
            for v in remaining_adj[u]:
                if u < v:
                    cliques.append(frozenset((u, v)))
                    
        return [list(c) for c in cliques]

    def local_search(cliques, adj, n, time_limit):
        """
        Improve the partition by local moves.
        Moves:
        1. Merge: If C1 U C2 is a clique, replace C1, C2 with C1 U C2.
        2. Pack: Pick a clique C, remove it, and try to repack its edges 
           into existing cliques or form new ones to potentially reduce count 
           (This is complex, so we focus on Merging and a simplified 'Shift' move).
        
        Actually, a more effective move for dense graphs is:
        - Identify "wasted" capacity. If two cliques share a common neighbor structure, 
          maybe they can be merged? No, merging requires the union to be a clique.
        
        Let's use a "Re-packing" heuristic:
        - Pick a clique C.
        - Remove C from the partition.
        - Try to insert edges of C into existing cliques to form larger cliques.
        - If we can form a new clique from a subset of C's edges and some edges from 
          other cliques, we might reduce the total count if we break up other cliques 
          efficiently.
        
        Simpler and more robust move: **Greedy Re-insertion**.
        1. Remove a random clique C.
        2. For each edge (u,v) in C, try to add it to an existing clique K 
           such that K U {(u,v)} is a clique. 
           (Note: K U {(u,v)} is a clique if u,v are connected to all in K and u-v is edge).
           This effectively "moves" edges from C to other cliques.
        3. If all edges of C are moved, C is removed (count decreases by 1).
        4. If some edges remain, they form a new clique (count stays same or increases).
        
        This is essentially a "move to reduce count" if we can fully absorb a clique.
        """
        
        cliques = [list(c) for c in cliques]
        # Index cliques by their edges for fast lookup? 
        # Or just keep list of lists.
        
        # Build a map: edge -> list of indices of cliques containing this edge
        # Since partition is disjoint, edge -> index of clique
        edge_to_clique = {}
        for idx, c in enumerate(cliques):
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    edge_to_clique[(c[i], c[j])] = idx
                    edge_to_clique[(c[j], c[i])] = idx

        improved = True
        last_improvement = time.time()
        
        while time.time() < time_limit and improved:
            improved = False
            # Shuffle cliques to avoid bias
            random.shuffle(cliques)
            
            # Rebuild edge_to_clique after shuffle
            edge_to_clique = {}
            for idx, c in enumerate(cliques):
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        edge_to_clique[(c[i], c[j])] = idx
                        edge_to_clique[(c[j], c[i])] = idx

            for idx in range(len(cliques)):
                if time.time() > time_limit:
                    break
                
                C = cliques[idx]
                if len(C) < 2:
                    continue
                    
                # Try to move all edges of C to other cliques
                # We simulate removing C
                # We need to check if each edge (u,v) in C can be added to some other clique K
                # such that K U {u,v} is a clique.
                
                # Optimization: Only try to move if C is "small" or "isolated"?
                # Let's just try.
                
                # We need to track which edges are "moved" to which new clique.
                # If we move edge (u,v) to clique K, the new clique is K U {u,v}.
                # Multiple edges from C might go to the same K.
                
                # This is a constraint satisfaction problem for the edges of C.
                # Given the small size of C (usually), we can try a simple greedy:
                # For each edge in C, find a compatible K.
                
                # To be safe and fast, we only attempt this if C is of size <= 5 
                # or if we have plenty of time.
                if len(C) > 6:
                    continue

                edges_of_C = []
                for i in range(len(C)):
                    for j in range(i+1, len(C)):
                        edges_of_C.append((C[i], C[j]))
                
                # For each edge, find a target clique
                # target[edge] = index of clique to move to
                # We must ensure that if we move multiple edges to the same target K,
                # the resulting set is still a clique.
                
                # Simplified Move: 
                # Find a clique K such that K U C is a clique.
                # If so, merge C into K. This reduces count by 1.
                
                # This is the "Merge" move. It's the most effective for reducing count.
                # Let's prioritize this.
                
                merged = False
                for k_idx in range(len(cliques)):
                    if k_idx == idx:
                        continue
                    K = cliques[k_idx]
                    # Check if K U C is a clique
                    # This is expensive if K and C are large.
                    # But we can check: 
                    # 1. K is clique (it is)
                    # 2. C is clique (it is)
                    # 3. Every node in K is connected to every node in C.
                    
                    valid = True
                    for u in K:
                        for v in C:
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Merge!
                    # New clique = K U C
                    new_clique = list(set(K) | set(C))
                    # Remove K and C
                    # We will modify the list in place
                    # Remove idx (C) and k_idx (K)
                    # Careful with indices if k_idx < idx
                    
                    if k_idx < idx:
                        # Remove k_idx first, then idx-1
                        cliques.pop(k_idx)
                        cliques.pop(idx - 1)
                    else:
                        cliques.pop(idx)
                        cliques.pop(k_idx)
                    
                    cliques.append(new_clique)
                    
                    # Rebuild edge map
                    edge_to_clique = {}
                    for idx_new, c_new in enumerate(cliques):
                        for i in range(len(c_new)):
                            for j in range(i+1, len(c_new)):
                                edge_to_clique[(c_new[i], c_new[j])] = idx_new
                                edge_to_clique[(c_new[j], c_new[i])] = idx_new
                                
                    improved = True
                    last_improvement = time.time()
                    merged = True
                    break
                
                if merged:
                    # Restart loop because list changed
                    break 

        return cliques

    def verify_and_fix(cliques, adj, n):
        """
        Ensure validity: 
        1. Every clique is a clique.
        2. Every edge is in exactly one clique.
        3. No empty cliques.
        Fixes any violations by splitting or removing.
        """
        valid_cliques = []
        covered_edges = set()
        
        for c in cliques:
            if len(c) < 2:
                continue
            if is_clique(c, adj):
                # Check for duplicate edges
                new_edges = set()
                ok = True
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        e = (c[i], c[j])
                        if e in covered_edges:
                            ok = False
                            break
                        new_edges.add(e)
                    if not ok:
                        break
                if ok:
                    valid_cliques.append(c)
                    covered_edges.update(new_edges)
        
        # Find uncovered edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        missing = all_edges - covered_edges
        
        # Add missing edges as 2-cliques
        for u, v in missing:
            valid_cliques.append([u, v])
            
        return valid_cliques

    def solve_instance(n, p, seed):
        adj = get_graph(n, p, seed)
        
        # 1. Greedy Initial Partition
        t_init = time.time()
        cliques = greedy_partition(adj, n)
        
        # 2. Local Search
        t_ls = time.time()
        time_left = deadline - t_ls
        if time_left > 5:
            cliques = local_search(cliques, adj, n, deadline - 5)
        
        # 3. Verify and Fix
        cliques = verify_and_fix(cliques, adj, n)
        
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 10:
             # Emergency fallback
             adj = get_graph(n, p, seed)
             # Just return all edges as 2-cliques
             cliques = []
             for u in range(n):
                 for v in adj[u]:
                     if u < v:
                         cliques.append([u, v])
             results.append(cliques)
             continue
             
        cliques = solve_instance(n, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
