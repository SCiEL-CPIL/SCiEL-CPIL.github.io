import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    def is_regular_subgraph(mask, adj_masks, n):
        """
        Check if the set of vertices in 'mask' induces a regular subgraph.
        adj_masks: list of int, where adj_masks[i] has bit j set if edge (i,j) exists.
        """
        # Get list of vertices in the subset
        verts = [i for i in range(n) if (mask >> i) & 1]
        k = len(verts)
        if k < 6:
            return False
        
        # Calculate degrees within the subset for each vertex
        degrees = []
        for v in verts:
            # Count neighbors of v that are also in the subset
            # adj_masks[v] & mask gives the neighbors of v within the subset
            # But we must exclude v itself if it's in the mask (it's not, simple graph)
            # However, adj_masks[v] contains v? No, simple graph, no self loops.
            # So popcount(adj_masks[v] & mask) is the degree of v in the induced subgraph.
            deg = bin(adj_masks[v] & mask).count('1')
            degrees.append(deg)
        
        # Check if all degrees are the same
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def count_bad_subgraphs(adj_masks, n, min_size=6):
        """
        Count the number of induced subgraphs of size >= min_size that are regular.
        This is expensive, so we only call it for verification or during search if n is small.
        For n=20, C(20,6) = 38760. This is fast enough to run every few steps? 
        38760 * 6 * popcount ~ 100k ops. In Python, maybe 0.05s. 
        If we do this every step, 100 steps = 5s. Acceptable.
        """
        count = 0
        # We primarily care about size 6. Larger regular subgraphs are rarer.
        # If we minimize size 6, we are likely to be valid.
        # Let's count size 6 only for speed in the loop, and verify full at the end.
        for S in itertools.combinations(range(n), min_size):
            mask = 0
            for v in S:
                mask |= (1 << v)
            if is_regular_subgraph(mask, adj_masks, n):
                count += 1
        return count

    def find_bad_subgraph(adj_masks, n, min_size=6):
        """
        Find one regular subgraph of size min_size.
        """
        for S in itertools.combinations(range(n), min_size):
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

    def random_graph(n):
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
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
    # 1. Try to find a valid graph for n=20. This is a known target.
    # 2. If time permits, try n=21.
    
    best_graph = None
    best_n = 0

    # Try n=20 first as it's safer
    for target_n in [20, 21]:
        if time.time() > deadline - 10:
            break
            
        if target_n == 21 and best_n < 20:
            # If we haven't found a 20-vertex graph, skip 21
            continue

        max_starts = 5 if target_n == 20 else 3
        max_steps = 200 if target_n == 20 else 100

        for start_idx in range(max_starts):
            if time.time() > deadline - 5:
                break
                
            adj_masks = random_graph(target_n)
            current_bad_count = count_bad_subgraphs(adj_masks, target_n, 6)
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                if current_bad_count == 0:
                    # Verify full graph (sizes > 6)
                    if verify_graph(adj_masks, target_n):
                        if target_n > best_n:
                            best_n = target_n
                            best_graph = adj_masks[:]
                        break
                    else:
                        # Found a larger regular subgraph, but no size 6.
                        # This is rare if we minimized size 6. 
                        # Try to flip an edge in the larger subgraph.
                        # For simplicity, if verify fails, we just continue searching 
                        # or accept if it's size 20 and we've tried enough.
                        # Actually, if count_bad(6)==0 but verify fails, we have a regular subgraph of size > 6.
                        # We should try to break it.
                        # Let's find it.
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
                            current_bad_count = count_bad_subgraphs(adj_masks, target_n, 6)
                            continue
                        else:
                            # Should not happen if verify failed
                            break

                # Find a bad size-6 subgraph to guide the flip
                bad_S = find_bad_subgraph(adj_masks, target_n, 6)
                if bad_S is None:
                    # No size 6 bad subgraphs, loop back to verify
                    continue

                # Heuristic: Flip an edge inside the bad subset.
                # To improve, we can try flipping each edge in the subset and pick the one that reduces bad count most?
                # That's expensive (6*5/2 = 15 edges * count_bad_subgraphs).
                # count_bad_subgraphs is ~0.05s. 15 * 0.05 = 0.75s per step. Too slow for 200 steps.
                # Random flip inside is faster.
                
                u, v = random.sample(bad_S, 2)
                flip_edge(adj_masks, u, v)
                current_bad_count = count_bad_subgraphs(adj_masks, target_n, 6)

    if best_graph is not None:
        return {'n': best_n, 'edges': to_edges(best_graph, best_n)}
    
    # Fallback: Return a small valid graph if all else fails
    # A graph with n=5 is trivially valid (no subset of size 6)
    return {'n': 5, 'edges': []}
