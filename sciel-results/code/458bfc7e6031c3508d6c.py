import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20
    
    # Helper: Check if graph is K3,3-free
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def count_k33s(adj_bits):
        """
        Count the number of K3,3 subgraphs.
        """
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    cnt = common.bit_count()
                    if cnt >= 3:
                        # Number of ways to choose 3 from cnt
                        total += cnt * (cnt - 1) * (cnt - 2) // 6
        return total // 2

    def energy(adj_bits):
        e = count_k33s(adj_bits)
        m = count_edges(adj_bits)
        return e * 1000 - m

    # Strategy: 
    # 1. Initialize with a random dense graph
    # 2. Use a faster local search that specifically targets edge addition/removal
    #    with an optimized validity check.
    
    # Initialize
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                add_edge(adj_bits, i, j)
                
    best_adj = adj_bits[:]
    best_edges = count_edges(adj_bits)
    
    current_adj = adj_bits[:]
    
    # Fast validity check for a specific move
    # Returns True if adding edge (u,v) keeps the graph K3,3-free
    def can_add_edge(adj_bits, u, v):
        # A K3,3 is formed if there exists a set of 3 vertices S such that
        # all vertices in S are connected to u, all vertices in S are connected to v,
        # and all vertices in S are connected to each other? No.
        # A K3,3 consists of two sets A and B of size 3, with all edges between A and B.
        # Adding edge (u,v) creates a new K3,3 if and only if there exists a partition
        # of some 6 vertices into A and B such that (u,v) is the only missing edge
        # in the complete bipartite graph between A and B?
        # Actually, if (u,v) is added, it completes a K3,3 if there are sets A, B
        # such that u in A, v in B (or vice versa), and all other edges between A and B exist.
        
        # More simply:
        # Check all triples containing u? No.
        # Check all triples T. If u and v are in N(T) (common neighbors of T),
        # and u and v are not connected to each other (they are, we are adding it),
        # then does T form one side of K3,3?
        
        # Let's use the property:
        # Adding (u,v) creates a K3,3 iff there exists a set of 3 vertices W
        # such that:
        # 1. u is connected to all w in W
        # 2. v is connected to all w in W
        # 3. The edges between {u,v} and W are the only ones needed?
        # No, a K3,3 has 3 vertices on one side and 3 on the other.
        
        # If we add (u,v), a new K3,3 must include this edge.
        # So {u} and {v} must be in different partitions.
        # Let A contain u and B contain v.
        # A has 3 vertices: {u, a1, a2}. B has 3 vertices: {v, b1, b2}.
        # All edges between A and B must exist.
        # Since (u,v) is the new edge, we need:
        # u-b1, u-b2 exist (u connected to b1, b2)
        # v-a1, v-a2 exist (v connected to a1, a2)
        # a1-v, a2-v exist (same as above)
        # a1-b1, a1-b2, a2-b1, a2-b2 exist (a1, a2 connected to b1, b2)
        # Also a1-a2, b1-b2 don't matter (within partition).
        # But wait, a1, a2 must be connected to u? Yes, u-a1, u-a2 exist.
        # And b1, b2 connected to v? Yes, v-b1, v-b2 exist.
        
        # So, we need to find a1, a2, b1, b2 such that:
        # 1. u is connected to a1, a2, b1, b2
        # 2. v is connected to a1, a2, b1, b2
        # 3. a1 is connected to b1, b2
        # 4. a2 is connected to b1, b2
        # 5. a1, a2 are distinct from u, v, b1, b2
        # 6. b1, b2 are distinct from u, v, a1, a2
        
        # This is still complex. Let's stick to the O(n^3) check for simplicity
        # but optimize the inner loop.
        
        # For each triple (i, j, k):
        # If i, j, k are all neighbors of u and all neighbors of v,
        # AND i, j, k are mutually connected? No, i, j, k form one side.
        # If i, j, k form one side (A), and {u, v, x} form the other side (B)?
        # If u, v, x are all connected to i, j, k.
        # Then {i, j, k} and {u, v, x} form a K3,3 if:
        # - u, v, x are distinct
        # - u, v, x are all connected to i, j, k
        # - (u,v) is the edge being added.
        
        # So, for a fixed triple (i, j, k):
        # Let C = common neighbors of i, j, k.
        # If u in C and v in C:
        #   We need a third vertex x in C such that x is connected to u and v?
        #   Wait, if x in C, x is connected to i, j, k.
        #   For {i, j, k} and {u, v, x} to be a K3,3, we need:
        #   - u, v, x distinct
        #   - u connected to v (yes, adding it)
        #   - u connected to x
        #   - v connected to x
        #   - i, j, k connected to u, v, x (yes, by definition of C)
        
        # So, check:
        # For each triple (i, j, k):
        #   C = N(i) & N(j) & N(k)
        #   If u in C and v in C:
        #       If there exists x in C \ {u, v} such that (u-x) and (v-x) exist:
        #           Then adding (u,v) creates a K3,3.
        
        # Complexity: O(n^3 * cost_of_intersection).
        # n=20, 8000 triples. Very fast.
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    C = adj_bits[i] & adj_bits[j] & adj_bits[k]
                    if (C >> u) & 1 and (C >> v) & 1:
                        # Check if there is an x in C (x != u, x != v) connected to both u and v
                        # x must be in C, and connected to u and v.
                        # Since x in C, x is connected to i, j, k.
                        # We need x connected to u and v.
                        # Let's check all x in C.
                        mask = C & ~((1 << u) | (1 << v))
                        while mask:
                            lsb = mask & -mask
                            x = lsb.bit_length() - 1
                            if ((adj_bits[u] >> x) & 1) and ((adj_bits[v] >> x) & 1):
                                return False
                            mask ^= lsb
        return True

    # Local search loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to add edges
        for i in range(n):
            for j in range(i+1, n):
                if not (current_adj[i] >> j) & 1:
                    if can_add_edge(current_adj, i, j):
                        add_edge(current_adj, i, j)
                        improved = True
                        best_adj = current_adj[:]
                        best_edges = count_edges(current_adj)
        
        # Try to remove edges and re-add others? 
        # If stuck, try removing an edge to unlock more additions.
        if not improved:
            # Find an edge to remove that might allow more additions
            # Heuristic: Remove edge with highest degree sum? Or random?
            # Let's try a few random removals
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                u, v = random.sample(range(n), 2)
                if (current_adj[u] >> v) & 1:
                    remove_edge(current_adj, u, v)
                    # Try to add edges
                    local_improved = True
                    while local_improved:
                        local_improved = False
                        for i in range(n):
                            for j in range(i+1, n):
                                if not (current_adj[i] >> j) & 1:
                                    if can_add_edge(current_adj, i, j):
                                        add_edge(current_adj, i, j)
                                        local_improved = True
                                        best_adj = current_adj[:]
                                        best_edges = count_edges(current_adj)
                    if count_edges(current_adj) > best_edges:
                        best_adj = current_adj[:]
                        best_edges = count_edges(current_adj)
                    # Revert the initial removal? 
                    # No, keep the state if it's better.
                    # But if it's worse, we might want to revert.
                    # For now, just keep exploring.

    # Final check
    final_adj = best_adj[:]
    if count_k33s(final_adj) > 0:
        # Repair
        for _ in range(1000):
            if count_k33s(final_adj) == 0:
                break
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = final_adj[i] & final_adj[j] & final_adj[k]
                        if common.bit_count() >= 3:
                            verts = []
                            mask = common
                            while len(verts) < 3 and mask:
                                lsb = mask & -mask
                                v = lsb.bit_length() - 1
                                verts.append(v)
                                mask ^= lsb
                            u, v = random.choice([(i,verts[0]), (j,verts[1]), (k,verts[2])])
                            remove_edge(final_adj, u, v)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break

    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (final_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result

# EVOLVE-BLOCK-END
