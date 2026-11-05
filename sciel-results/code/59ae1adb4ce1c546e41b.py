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

    def get_bits(mask):
        """Get list of set bit indices in a mask."""
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits

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

    def merge_cliques(n, partition, edges_set):
        """
        Attempt to merge cliques in the partition if their union is also a clique.
        This is a post-processing step to reduce the total number of cliques.
        """
        # Build adjacency bitmask for the original graph for quick clique checks
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Convert partition to a list of sets for easier manipulation
        cliques = [set(c) for c in partition]
        
        # We will iteratively try to merge cliques
        # To make it efficient, we can sort by size or just iterate
        # A simple approach: repeatedly try to merge any two cliques
        # This can be O(k^2) where k is number of cliques, but k is small for dense graphs
        # Let's limit the number of passes to avoid infinite loops or excessive time
        max_passes = 10
        changed = True
        passes = 0
        
        while changed and passes < max_passes:
            changed = False
            passes += 1
            if time.time() - start_time > DEADLINE:
                break
                
            # Try to merge cliques[i] and cliques[j]
            # We iterate through all pairs. If a merge happens, we restart the scan or continue carefully.
            # Since merging reduces the number of cliques, the list shrinks.
            # To be safe and simple, we can do a pass where we try to merge the first mergeable pair, then restart.
            # Or, we can do a full scan and merge if possible.
            
            merged_any = False
            i = 0
            while i < len(cliques) and not merged_any:
                j = i + 1
                while j < len(cliques) and not merged_any:
                    # Check if union of cliques[i] and cliques[j] is a clique
                    union = cliques[i] | cliques[j]
                    if len(union) < 2:
                        j += 1
                        continue
                    
                    # Check if union is a clique in the original graph
                    is_clique = True
                    # Efficient check using bitmasks
                    # For each vertex in union, check if all other vertices in union are neighbors
                    # Or simply: for all pairs in union, check edge existence
                    # Since union is small (usually), pair check is fine.
                    # Let's use bitmask for speed if union is large?
                    # For now, simple pair check.
                    u_list = list(union)
                    for a in range(len(u_list)):
                        for b in range(a + 1, len(u_list)):
                            u = u_list[a]
                            v = u_list[b]
                            if not (adj[u] & (1 << v)):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        # Merge
                        cliques[i] = union
                        cliques.pop(j)
                        changed = True
                        merged_any = True
                        # Restart scan from beginning as indices have shifted
                        i = 0
                    else:
                        j += 1
                if not merged_any:
                    i += 1
            
            if not changed:
                break

        return [sorted(c) for c in cliques]

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        partition = greedy_clique_partition_bitmask(n, edges_set)
        # Apply merge post-processing
        partition = merge_cliques(n, partition, edges_set)
        return partition

    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance: just list all edges as 2-cliques
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]

    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
