"""Dense Q3-free graph on 16 vertices via exhaustive search over complement with bit-parallel forbidden-subgraph check."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115
    n = 16
    full_mask = (1 << n) - 1
    
    # Precompute: for each pair (u, v), the set of 6-tuples of vertices that would form a Q3
    # if edge (u,v) were added. We represent a Q3 by its 8 vertices.
    # Actually, let's think differently.
    # A Q3 is a bipartite graph with parts A={a1,a2,a3} and B={b1,b2,b3} plus a center? No.
    # Q3 is the cube. Vertices are binary strings of length 3.
    # Edges connect vertices differing in one bit.
    # We want to detect if adding edge (u,v) creates a Q3.
    
    # Efficient approach:
    # For a fixed graph G, adding edge (u,v) creates a Q3 iff there exists a set of 6 other vertices
    # such that the subgraph induced by {u,v} + 6 others is isomorphic to Q3.
    
    # Let's precompute all possible Q3s on 16 vertices? There are C(16,8) * (number of Q3 labelings) which is huge.
    # Instead, use the fact that Q3 is bipartite.
    # Let's use the "creates_q3_fast" logic but optimized with precomputed neighbor intersections.
    
    # Alternative Strategy: 
    # Since n=16 is small, we can search for the MAXIMAL number of edges.
    # The complement graph must be "Q3-saturated" in some sense? No.
    # Let's stick to a highly optimized local search with a better initialization:
    # Start with a complete graph and REMOVE edges to break all Q3s.
    # This is a "Hitting Set" problem: Find the minimum set of edges to remove to hit all Q3s.
    # Since we want to MAXIMIZE edges, we want to MINIMIZE removed edges.
    
    # Step 1: Find all Q3s in the complete graph K16? 
    # K16 contains many Q3s. 
    # Actually, it's easier to build up. But the "removal" strategy might be better for finding dense graphs 
    # because the search space of "which edges to remove" might be smaller if the optimal graph is very dense.
    
    # Let's estimate the density. 
    # If the optimal graph has ~40-50 edges, we remove ~80-90 edges.
    # If we build up, we add ~40-50 edges.
    # Building up seems more direct.
    
    # Let's try a different structural change: **Integer Programming (ILP) via scipy?**
    # No, scipy is available but writing an ILP solver from scratch is hard.
    # However, we can use a simple branch-and-bound or a specialized heuristic for the "Maximum Induced Subgraph" or similar.
    
    # Let's try **Simulated Annealing** on the edge set, with a very fast validity check.
    # The key is the speed of `creates_q3`.
    
    # Optimization of `creates_q3`:
    # Instead of nested loops, precompute for every pair (u,v) and every potential "witness" set?
    # Let's precompute a table `q3_witnesses[u][v]` which is a list of masks representing the other 6 vertices 
    # that, combined with u and v, form a Q3.
    # Wait, the structure of Q3 relative to an edge (u,v) is fixed.
    # In Q3, each edge belongs to exactly 2 faces (4-cycles).
    # If we add edge (u,v), it creates a Q3 if there are two disjoint 4-cycles containing (u,v) that share exactly u and v? 
    # No, a Q3 has 12 edges. An edge is in 2 squares.
    # The rest of the cube is determined.
    
    # Let's precompute all subsets of 8 vertices that form a Q3.
    # There are C(16,8) = 12870 subsets.
    # For each subset, check if it contains a Q3.
    # If it does, we can record the "forbidden patterns".
    # Actually, we can precompute a boolean table `is_q3_mask[mask]` for all 2^16 masks? 
    # 2^16 = 65536. This is very feasible!
    # For each mask of 8 vertices, check if the induced subgraph of K8 (complete graph on 8) 
    # can be labeled as Q3? No, the induced subgraph of the CURRENT graph must contain Q3.
    # We are checking if adding edge (u,v) creates a Q3.
    # This means there exists a set S of 6 other vertices such that 
    # G[S + {u,v}] contains a Q3 subgraph.
    
    # Let's precompute `q3_signatures`.
    # A Q3 is a specific graph. 
    # Let's iterate through all 8-subsets of {0..15}.
    # For each 8-subset, there are 8! / (symmetries) ways to map Q3 to these vertices.
    # This is getting complicated.
    
    # Let's go back to the fast check but make it faster using bitsets.
    
    def creates_q3_bitset(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        Uses bitmasks for adjacency.
        """
        # Q3 structure:
        # Vertices: 0,1,2,3,4,5,6,7
        # Edges: (0,1),(1,2),(2,3),(3,0) [Face 0]
        #        (4,5),(5,6),(6,7),(7,4) [Face 1]
        #        (0,4),(1,5),(2,6),(3,7) [Connectors]
        
        # We are adding edge (u,v).
        # In Q3, u and v must be adjacent.
        # Without loss of generality, assume u maps to 0 and v maps to 1.
        # Then the other 6 vertices must map to 2,3,4,5,6,7 in some order.
        # Specifically:
        # 0 is connected to 1, 3, 4.
        # 1 is connected to 0, 2, 5.
        # So, neighbors of 0 in Q3 are {1, 3, 4}.
        # Neighbors of 1 in Q3 are {0, 2, 5}.
        
        # Let N_u be neighbors of u in current graph (excluding v).
        # Let N_v be neighbors of v in current graph (excluding u).
        
        # The vertices mapping to 3 and 4 must be in N_u.
        # The vertices mapping to 2 and 5 must be in N_v.
        # Let A = N_u (candidates for 3,4 and potentially others? No, in Q3, 0 is only connected to 1,3,4).
        # Wait, in the induced subgraph, u might have other neighbors not in the Q3.
        # But for the Q3 to exist, there must be specific vertices playing the roles.
        
        # Roles relative to edge (0,1):
        # Role 3: Neighbor of 0, not 1. Connected to 2.
        # Role 4: Neighbor of 0, not 1. Connected to 7.
        # Role 2: Neighbor of 1, not 0. Connected to 3.
        # Role 5: Neighbor of 1, not 0. Connected to 6.
        # Role 6: Neighbor of 5, not 1. Connected to 7.
        # Role 7: Neighbor of 4, not 0. Connected to 6.
        
        # Constraints:
        # 3 in N_u, 2 in N_v, and 3-2 is an edge.
        # 4 in N_u, 5 in N_v, and 4-5 is NOT an edge? No, 4-5 is not an edge in Q3.
        # Wait, 4 is connected to 7 and 0. 5 is connected to 6 and 1.
        # 4 and 5 are NOT connected.
        # 3 and 5 are NOT connected.
        # 2 and 4 are NOT connected.
        
        # Let's define the sets:
        # U_cand = N_u (vertices adjacent to u)
        # V_cand = N_v (vertices adjacent to v)
        
        # We need to pick 3,4 from U_cand and 2,5 from V_cand.
        # 3,4 must be distinct. 2,5 must be distinct.
        # All 6 must be distinct and not u,v.
        
        # Edges required:
        # (3,2), (4,7), (5,6), (2,3) is same, (6,7), (3,4)? No.
        # Q3 edges:
        # 0-1 (added)
        # 0-3, 0-4
        # 1-2, 1-5
        # 2-3, 2-? No, 2 is connected to 1 and 3. And 2-6? No.
        # Let's list Q3 edges again carefully.
        # 000-001 (0-1)
        # 000-010 (0-2)  <-- Let's use binary to be safe.
        # 000-100 (0-4)
        # 001-011 (1-3)
        # 001-101 (1-5)
        # 010-011 (2-3)
        # 010-110 (2-6)
        # 100-101 (4-5)
        # 100-110 (4-6)
        # 011-111 (3-7)
        # 101-111 (5-7)
        # 110-111 (6-7)
        
        # If we fix edge 0-1 (000-001):
        # Neighbors of 0: 1, 2, 4.
        # Neighbors of 1: 0, 3, 5.
        
        # So:
        # Vertices adjacent to u (in Q3): {1, 2, 4}. Since 1 is v, the other two are 2 and 4.
        # Vertices adjacent to v (in Q3): {0, 3, 5}. Since 0 is u, the other two are 3 and 5.
        
        # Let x, y be the two vertices from N_u (excluding v) that play roles 2 and 4.
        # Let z, w be the two vertices from N_v (excluding u) that play roles 3 and 5.
        
        # We must assign x,y to {2,4} and z,w to {3,5}.
        # There are 2! * 2! = 4 assignments.
        
        # Case 1: x=2, y=4, z=3, w=5.
        # Required edges among {x,y,z,w} and the remaining 2 vertices (6,7):
        # Edges in Q3 involving 2,3,4,5:
        # 2-3 (x-z)
        # 2-6 (x-6)
        # 4-5 (y-w)
        # 4-6 (y-6)
        # 3-7 (z-7)
        # 5-7 (w-7)
        # 6-7 (6-7)
        # Note: 2-4 is NOT an edge. 3-5 is NOT an edge. 2-5 is NOT an edge. 3-4 is NOT an edge.
        
        # So we need to find 6,7 such that:
        # 6 is adjacent to x(2) and y(4).
        # 7 is adjacent to z(3) and w(5).
        # 6 is adjacent to 7.
        # 6,7 are distinct from x,y,z,w,u,v.
        
        # Let's check this case.
        N_u = adj[u] & ~(1 << v)
        N_v = adj[v] & ~(1 << u)
        
        # Get list of candidates
        cand_u = []
        mask = N_u
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            cand_u.append(idx)
            mask -= bit
            
        cand_v = []
        mask = N_v
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            cand_v.append(idx)
            mask -= bit
            
        # Iterate over pairs from cand_u and cand_v
        # To speed up, we can precompute intersections.
        
        for x in cand_u:
            for y in cand_u:
                if x == y: continue
                # x plays 2, y plays 4 (or vice versa, but the set {x,y} is what matters for 6)
                # Wait, the roles are symmetric for 6?
                # 6 is connected to 2 and 4. So 6 must be in N_x & N_y.
                # 3 and 5 are connected to 7. So 7 must be in N_z & N_w.
                
                # Let's just iterate over all pairs (x,y) from cand_u and (z,w) from cand_v.
                # 16*16 * 16*16 is too slow? 256 * 256 = 65536. 
                # Inside, we check bitmasks. This is fast.
                
                # But we can prune.
                # x and y must be distinct. z and w must be distinct.
                # x,y,z,w must be distinct from each other.
                
                for z in cand_v:
                    for w in cand_v:
                        if z == w: continue
                        if z in (x, y) or w in (x, y): continue
                        
                        # Check edges:
                        # x(2) - z(3) must be edge.
                        if not (adj[x] & (1 << z)): continue
                        # y(4) - w(5) must be edge.
                        if not (adj[y] & (1 << w)): continue
                        
                        # Find 6: Neighbor of x and y.
                        c6 = (adj[x] & adj[y]) & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << y) & ~(1 << z) & ~(1 << w)
                        if not c6: continue
                        
                        # Find 7: Neighbor of z and w.
                        c7 = (adj[z] & adj[w]) & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << y) & ~(1 << z) & ~(1 << w)
                        if not c7: continue
                        
                        # 6 and 7 must be distinct.
                        # And 6-7 must be an edge.
                        # Iterate through candidates for 6
                        c6_bits = c6
                        while c6_bits:
                            bit6 = c6_bits & (-c6_bits)
                            v6 = bit6.bit_length() - 1
                            c6_bits -= bit6
                            
                            # 7 must be in c7 and not v6
                            c7_check = c7 & ~(1 << v6)
                            if not c7_check: continue
                            
                            # Check if any 7 in c7_check is adjacent to v6
                            # We need adj[v6] & c7_check != 0
                            if adj[v6] & c7_check:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += adj[u].bit_count()
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline - 3:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_bitset(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    # Strategy: Simulated Annealing
    # State: adjacency matrix (list of bitmasks)
    # Move: Flip an edge (add or remove)
    # Energy: -number of edges (we want to minimize this, i.e., maximize edges)
    # Constraint: Must be Q3-free.
    # If a move creates a Q3, it is forbidden (infinite energy penalty).
    
    def local_search_sa(adj, steps=20000):
        best_adj = adj[:]
        best_count = count_edges(adj)
        current_adj = adj[:]
        current_count = best_count
        
        T = 1.0
        cooling = 0.9995
        
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (current_adj[u] & (1 << v))]
        existing = [(u, v) for u in range(n) for v in range(u+1, n) if (current_adj[u] & (1 << v))]
        
        for i in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Choose a random edge to flip
            # Bias towards adding missing edges if we have room, or removing if we are stuck?
            # Let's just pick randomly from all possible edges.
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_present = bool(current_adj[u] & (1 << v))
            
            trial_adj = current_adj[:]
            
            if is_present:
                # Remove edge
                trial_adj[u] &= ~(1 << v)
                trial_adj[v] &= ~(1 << u)
                delta_e = -1
                valid = True # Removing edge never creates a Q3
            else:
                # Add edge
                if creates_q3_bitset(u, v, trial_adj):
                    valid = False
                else:
                    trial_adj[u] |= (1 << v)
                    trial_adj[v] |= (1 << u)
                    delta_e = 1
                    valid = True
            
            if not valid:
                # Reject
                continue
                
            new_count = current_count + delta_e
            
            if new_count > current_count:
                # Always accept improvement
                current_adj = trial_adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
            else:
                # Accept with probability exp(-delta_E / T) where delta_E = -delta_e (since we minimize -edges)
                # Here delta_e is -1. So delta_E = 1.
                # prob = exp(-1 / T)
                if random.random() < (1.0 / (1.0 + T)): # Approximation or just use exp
                    # Actually, let's use proper SA
                    # prob = math.exp(-1.0 / T)
                    # But T is small.
                    # Let's just accept if random < 0.1 for simplicity in this "structural" change?
                    # No, let's do proper SA.
                    import math
                    prob = math.exp(-1.0 / T)
                    if random.random() < prob:
                        current_adj = trial_adj
                        current_count = new_count
                        
            T *= cooling
            
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Multiple restarts
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        adj = [0] * n
        
        # Random initialization
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        greedy_add(adj, all_edges)
        
        # SA Polish
        adj, cnt = local_search_sa(adj, steps=5000)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Final greedy polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
