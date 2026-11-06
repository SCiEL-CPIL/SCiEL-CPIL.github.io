"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    n = 16
    # Q3 has 8 vertices, 12 edges
    # Q3 adjacency: vertices 0-7, edges between i and j if Hamming distance = 1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Precompute: for each pair of vertices, which Q3 edges they could correspond to
    # Actually, let's use a different approach: 
    # A Q3 subgraph exists if there's an injective mapping f: {0..7} -> {0..15} 
    # such that for all (i,j) in q3_edges, edge (f(i), f(j)) exists in G.
    
    # Better approach: use bitmask representation
    # adj[v] is a bitmask of neighbors of v
    # Start with complete graph
    adj = [0] * n
    for v in range(n):
        for u in range(n):
            if u != v:
                adj[v] |= (1 << u)
    
    def find_q3(adj):
        """Find a Q3 subgraph. Returns the 8 vertices or None."""
        # Try all 8-vertex subsets - too many (C(16,8)=12870)
        # For each, check if it contains Q3 as subgraph
        # This is expensive. Let's use a smarter approach.
        
        # Alternative: for each 8-vertex set, check if the induced subgraph 
        # contains Q3. But we need subgraph, not induced.
        
        # Actually, let's just check all 8-sets and see if Q3 can be embedded.
        # For an 8-set S, we need to check if there's a bijection from Q3 vertices to S
        # such that all Q3 edges are present.
        
        # This is a constraint satisfaction problem on 8 vertices.
        # We can use backtracking.
        
        for combo in combinations(range(n), 8):
            if time.time() - start_time > deadline - 5:
                return None
            # Check if Q3 can be embedded in this 8-set
            # Use backtracking: assign Q3 vertices 0-7 to combo vertices
            # q3_adj[i] = set of Q3 neighbors of i
            q3_adj = [set() for _ in range(8)]
            for i, j in q3_edges:
                q3_adj[i].add(j)
                q3_adj[j].add(i)
            
            # Try all permutations of combo for Q3 vertices 0-7
            # 8! = 40320 per combo, too slow for 12870 combos
            # Use backtracking with pruning
            assign = [-1] * 8  # q3_vertex -> combo_index
            combo_set = set(combo)
            
            def backtrack(q3v):
                if time.time() - start_time > deadline - 5:
                    return False
                if q3v == 8:
                    return True
                for ci in range(8):
                    if assign[ci] != -1:
                        continue
                    v = combo[ci]
                    # Check compatibility with already assigned Q3 vertices
                    ok = True
                    for q3u in range(q3v):
                        if assign[q3u] == -1:
                            continue
                        u = combo[assign[q3u]]
                        edge_exists = (adj[v] >> u) & 1
                        if q3u in q3_adj[q3v]:
                            if not edge_exists:
                                ok = False
                                break
                        else:
                            # No constraint - non-edge in Q3 can be edge or non-edge in G
                            pass
                    if ok:
                        assign[q3v] = ci
                        if backtrack(q3v + 1):
                            return True
                        assign[q3v] = -1
                return False
            
            assign = [-1] * 8
            if backtrack(0):
                # Found Q3
                return [combo[assign[i]] for i in range(8)]
        
        return None
    
    # Iteratively remove edges to break Q3 subgraphs
    max_iter = 500
    for it in range(max_iter):
        if time.time() - start_time > deadline - 10:
            break
        q3 = find_q3(adj)
        if q3 is None:
            break
        # Remove a random edge from the Q3
        # Map back to original vertices
        q3_verts = q3
        # Find edges in G among q3_verts that correspond to Q3 edges
        # For the found assignment, we know which Q3 vertices map to which
        # But we lost the assignment. Let's just pick a random edge among the 8 vertices.
        edges_among = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = q3_verts[i], q3_verts[j]
                if (adj[u] >> v) & 1:
                    edges_among.append((u, v))
        if not edges_among:
            break
        u, v = random.choice(edges_among)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Count edges and build result
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
