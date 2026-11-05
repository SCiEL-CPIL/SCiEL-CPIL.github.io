import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def popcount(x):
        """Count set bits in an integer."""
        return bin(x).count("1")

    def greedy_clique_partition_bitmask(n, edges_set):
        """
        Find a clique partition using bitmask-based greedy maximal clique extraction.
        """
        if not edges_set:
            return []

        # rem_adj[i] is a bitmask where bit j is 1 if edge (i, j) is still remaining.
        rem_adj = [0] * n
        for u, v in edges_set:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)

        partition = []

        # Helper to check if any edges remain
        def has_edges():
            for i in range(n):
                if rem_adj[i]:
                    return True
            return False

        while has_edges():
            if time.time() - start_time > DEADLINE:
                # Fallback: extract remaining edges as 2-cliques
                # This is slow but ensures termination
                # We need to find which edges are left
                # Iterate through all pairs
                for u in range(n):
                    mask = rem_adj[u]
                    while mask:
                        # Get lowest set bit
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        partition.append([u, v])
                        # Remove edge (u, v)
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                        mask &= mask - 1
                break

            # 1. Find vertex v with maximum degree in remaining graph
            max_deg = -1
            v = -1
            for i in range(n):
                deg = popcount(rem_adj[i])
                if deg > max_deg:
                    max_deg = deg
                    v = i
            
            if v == -1:
                break

            # 2. Build clique starting with v
            clique = [v]
            # Candidates are neighbors of v in the remaining graph
            candidates = rem_adj[v]
            
            while candidates:
                # Find u in candidates with max degree within candidates
                # degree of u in candidates = popcount(candidates & rem_adj[u])
                best_u = -1
                best_deg = -1
                
                # Iterate through set bits in candidates
                c_mask = candidates
                while c_mask:
                    lsb = c_mask & -c_mask
                    u = lsb.bit_length() - 1
                    # Calculate degree of u within the current candidate set
                    deg_u = popcount(candidates & rem_adj[u])
                    if deg_u > best_deg:
                        best_deg = deg_u
                        best_u = u
                    c_mask &= c_mask - 1
                
                if best_u == -1:
                    break
                
                clique.append(best_u)
                # Update candidates: must be neighbors of best_u
                candidates = candidates & rem_adj[best_u]

            # 3. Add clique to partition
            partition.append(sorted(clique))

            # 4. Remove all edges within the clique from rem_adj
            for i in range(len(clique)):
                ui = clique[i]
                for j in range(i + 1, len(clique)):
                    uj = clique[j]
                    rem_adj[ui] &= ~(1 << uj)
                    rem_adj[uj] &= ~(1 << ui)

        return partition

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        return greedy_clique_partition_bitmask(n, edges_set)

    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance: just list all edges as 2-cliques
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]

    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
