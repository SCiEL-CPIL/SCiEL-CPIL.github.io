import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # Helper functions using bitmasks for speed

    def is_regular_subgraph(mask, adj_masks, n):
        """
        Check if the set of vertices in 'mask' induces a regular subgraph.
        adj_masks: list of int, where adj_masks[i] has bit j set if edge (i,j) exists.
        """
        verts = [i for i in range(n) if (mask >> i) & 1]
        k = len(verts)
        if k < 6:
            return False
        
        degrees = []
        for v in verts:
            deg = (adj_masks[v] & mask).bit_count()
            degrees.append(deg)
        
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def count_bad_subgraphs_size6(adj_masks, n):
        """
        Count the number of induced subgraphs of size 6 that are regular.
        """
        count = 0
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            if is_regular_subgraph(mask, adj_masks, n):
                count += 1
        return count

    def find_bad_subgraph_size6(adj_masks, n):
        """
        Find one regular subgraph of size 6.
        """
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            if is_regular_subgraph(mask, adj_masks, n):
                return S
        return None

    def flip_edge(adj_masks, u, v):
        """
        Flip edge (u, v) in adj_masks.
        """
        adj_masks[u] ^= (1 << v)
        adj_masks[v] ^= (1 << u)

    def random_graph(n, p=0.5):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        return adj_masks

    def verify_graph(adj_masks, n):
        """
        Check if the graph has ANY regular induced subgraph of size >= 6.
        """
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                if is_regular_subgraph(mask, adj_masks, n):
                    return False
        return True

    def to_edges(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Strategy:
    # 1. Try to find a valid graph for n=21 first (the target for improvement).
    # 2. If time permits or n=21 fails, try n=20.
    # 3. Use a local search (hill climbing) that minimizes the number of regular
    #    induced subgraphs of size 6.
    # 4. Periodically verify the full graph (sizes >= 6) if size-6 count is 0.

    best_graph = None
    best_n = 0

    # Try n=21 first
    target_n_list = [21, 20, 19]
    
    for target_n in target_n_list:
        if time.time() > deadline - 10:
            break
            
        # If we already have a graph with n >= target_n, we can skip or just try to improve
        if best_n >= target_n:
            continue

        max_starts = 5 if target_n <= 20 else 10
        max_steps = 300 if target_n <= 20 else 150

        for start_idx in range(max_starts):
            if time.time() > deadline - 5:
                break
                
            adj_masks = random_graph(target_n)
            current_bad_count = count_bad_subgraphs_size6(adj_masks, target_n)
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                if current_bad_count == 0:
                    # Verify full graph (sizes > 6)
                    if verify_graph(adj_masks, target_n):
                        if target_n > best_n:
                            best_n = target_n
                            best_graph = adj_masks[:]
                        # If we found a valid 21, we can stop immediately as it's the max possible improvement
                        if target_n == 21:
                            return {'n': best_n, 'edges': to_edges(best_graph, best_n)}
                        break
                    else:
                        # Found a larger regular subgraph, but no size 6.
                        # Try to break it.
                        bad_S = None
                        for k in range(7, target_n + 1):
                            for S in itertools.combinations(range(target_n), k):
                                mask = 0
                                for v in S:
                                    mask |= (1 << v)
                                if is_regular_subgraph(mask, adj_masks, target_n):
                                    bad_S = S
                                    break
                            if bad_S:
                                break
                        if bad_S:
                            u, v = random.sample(bad_S, 2)
                            flip_edge(adj_masks, u, v)
                            current_bad_count = count_bad_subgraphs_size6(adj_masks, target_n)
                            continue
                        else:
                            break

                # Find a bad size-6 subgraph to guide the flip
                bad_S = find_bad_subgraph_size6(adj_masks, target_n)
                if bad_S is None:
                    # No size 6 bad subgraphs, loop back to verify
                    continue

                # Heuristic: Flip an edge inside the bad subset.
                # To improve efficiency, we can try flipping each edge in the subset 
                # and pick the one that reduces bad count most, but that's expensive.
                # Random flip inside is faster.
                
                u, v = random.sample(bad_S, 2)
                flip_edge(adj_masks, u, v)
                current_bad_count = count_bad_subgraphs_size6(adj_masks, target_n)

    if best_graph is not None:
        return {'n': best_n, 'edges': to_edges(best_graph, best_n)}
    
    # Fallback: Return a small valid graph if all else fails
    # A graph with n=5 is trivially valid (no subset of size 6)
    return {'n': 5, 'edges': []}

