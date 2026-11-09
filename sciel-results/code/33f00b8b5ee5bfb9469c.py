"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def solve_instance(n, p, seed, time_budget):
        edges, adj = instance(n, p, seed)
        if not edges:
            return []
        
        # Initial partition: Each edge is its own clique
        # This is a valid partition (every edge in exactly one clique, all cliques size 2)
        # We represent a clique as a frozenset of vertices.
        # We need to track which edges are in which clique.
        
        # For efficiency, we will work with lists of vertices for each clique.
        # Initial state: list of [u, v] for each edge (u, v)
        cliques = [[u, v] for u, v in edges]
        
        # To speed up checks, maintain a set of edges covered by each clique?
        # Or better: maintain a global set of all edges, and for each clique, the set of edges it covers.
        # However, merging requires checking if the union is a clique.
        
        # Let's use a different representation for the search:
        # We maintain a list of cliques (lists of vertices).
        # We maintain a dictionary: edge -> index of clique containing it.
        # This allows O(1) lookup of which clique an edge belongs to.
        
        edge_to_clique_idx = {}
        for i, c in enumerate(cliques):
            for u in c:
                for v in c:
                    if u < v:
                        edge_to_clique_idx[(u, v)] = i

        def is_clique(vertices, adj):
            vs = list(vertices)
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        return False
            return True

        def get_clique_edges(clique):
            es = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    es.add((min(u, v), max(u, v)))
            return es

        current_cost = len(cliques)
        
        # Local Search: Try to merge cliques or move edges.
        # Strategy: 
        # 1. Identify pairs of cliques that can be merged (union is a clique).
        # 2. If merged, reduce count by 1.
        # 3. If no merge possible, try to move an edge from a large clique to a smaller one 
        #    if it maintains validity and allows a future merge? 
        #    Actually, moving an edge usually increases cost or keeps it same if we split.
        #    Merging is the primary driver for reducing count.
        
        # A more powerful move: "Exchange"
        # Take an edge e from clique C1 and move it to clique C2 if C2+e is a clique.
        # This doesn't change cost immediately, but might enable C1 (minus e) to merge with C3.
        
        # Given time constraints, let's focus on aggressive Merging first, then simple local optimization.
        
        # Precompute adjacency matrix for fast lookup? 
        # adj is list of sets. O(1) lookup.
        
        merged = True
        while merged and time.time() < time_budget:
            merged = False
            # Shuffle cliques to avoid order bias
            indices = list(range(len(cliques)))
            random.shuffle(indices)
            
            for i_idx in indices:
                if time.time() > time_budget: break
                i = i_idx
                if len(cliques) <= i: continue
                c1 = cliques[i]
                if not c1: continue
                
                # Try to merge c1 with any other clique c2
                # Optimization: Only try to merge with cliques that share vertices?
                # No, disjoint cliques can't merge unless the graph is disconnected, which it isn't.
                # But they don't need to share vertices to be checked, but if they are disjoint, 
                # the union is a clique only if every vertex in c1 connects to every vertex in c2.
                
                # Let's collect candidate cliques to merge with c1.
                # A clique c2 is a candidate if c1 U c2 is a clique.
                
                # To speed this up, we can check connectivity.
                # If c1 and c2 are disjoint, we need |c1|*|c2| edge checks.
                # If they share vertices, we need fewer checks.
                
                # Let's just iterate through other cliques.
                for j_idx in indices:
                    if time.time() > time_budget: break
                    j = j_idx
                    if j == i: continue
                    if len(cliques) <= j: continue
                    c2 = cliques[j]
                    if not c2: continue
                    
                    # Check if c1 U c2 is a clique
                    union = set(c1) | set(c2)
                    valid = True
                    # Only need to check edges between c1 and c2
                    for u in c1:
                        for v in c2:
                            if u == v: continue
                            if v not in adj[u]:
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        # Merge c2 into c1
                        # Update edge_to_clique_idx
                        # Remove c2 from list, mark as empty or pop
                        # For simplicity, mark c2 as empty and skip in future iterations?
                        # Or better, rebuild the structure?
                        # Rebuilding is O(E). Merging is rare.
                        # Let's do a semi-in-place update.
                        
                        # Move edges from c2 to c1
                        for e in get_clique_edges(c2):
                            edge_to_clique_idx[e] = i
                        
                        # Add vertices from c2 to c1
                        c1.extend([v for v in c2 if v not in c1])
                        
                        # Mark c2 as empty
                        cliques[j] = []
                        
                        current_cost -= 1
                        merged = True
                        # Break inner loop to restart outer with new state?
                        # Or continue? Continuing might find more merges with the new c1.
                        # Let's continue to try more merges with the updated c1.
                        pass
            
        # After merging, we might have empty cliques. Filter them.
        final_cliques = [c for c in cliques if c]
        
        # Optional: Local Search to further reduce?
        # The merging process above is a form of greedy set cover (maximizing size).
        # Can we do better? 
        # Sometimes merging two small cliques prevents merging with a third.
        # A "swap" move could help.
        
        # Let's try a simple swap heuristic if time remains.
        # Pick a clique C, pick an edge e in C.
        # Pick another clique D.
        # If (C - e) U D is a clique, and D U e is a clique? No, that's complex.
        
        # Alternative: Random restarts with different merge orders?
        # The current approach is deterministic after shuffle.
        
        return final_cliques

    # Allocate time budget
    # Instance 1: n=32, Instance 2: n=64
    # n=64 is much harder (more edges, more cliques).
    # Give more time to the harder instance?
    # Or split equally?
    # Let's give 50% to first, 50% to second, with a small overhead reserve.
    
    t1_limit = start_time + 60
    t2_limit = deadline
    
    result1 = solve_instance(32, 0.6, 601, t1_limit)
    
    # If we finished early, use remaining time for result2
    # If we used all time, result2 gets minimal time
    result2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
