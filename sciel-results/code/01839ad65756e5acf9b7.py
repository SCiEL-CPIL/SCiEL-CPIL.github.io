import time
import itertools
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin

    def get_best_hypergraph(n, max_iters):
        """
        Finds a non-2-colorable 5-uniform hypergraph on n vertices
        minimizing edges using Set Cover formulation.
        """
        if time.time() > deadline:
            return None, float('inf')
            
        # 1. Generate all 5-edges
        edges = list(itertools.combinations(range(n), 5))
        m_edges = len(edges)
        
        # 2. Precompute masks for each edge
        # Universe: colorings of vertices 1..n-1 (vertex 0 fixed to 0)
        # Total colorings: 2^(n-1)
        num_colorings = 1 << (n - 1)
        
        # If n is large, this might be memory heavy or slow.
        # n=14 -> 8192 colorings. 
        # n=15 -> 16384.
        # n=16 -> 32768.
        
        # We represent a coloring as an integer 0..num_colorings-1
        # where bit i of the integer corresponds to vertex i+1.
        # Vertex 0 is always 0.
        
        # Precompute masks for each edge
        # mask_e is an integer where bit k is 1 if coloring k is monochromatic on e.
        
        # Vectorized precomputation using NumPy
        # Create array of all colorings
        # colors[i, v] = (i >> (v-1)) & 1 for v in 1..n-1
        
        # To save memory and time, we can process edges in batches or just loop if n is small.
        # For n=14, 2002 edges * 8192 colorings is fine.
        
        masks = []
        
        # Generate all colorings as a numpy array of shape (num_colorings, n)
        # But we only need bits for vertices 0..n-1.
        # Actually, we can just use bit operations in Python for the mask generation if it's fast enough,
        # or use NumPy for speed.
        
        # Let's use NumPy for speed.
        # indices = np.arange(num_colorings)
        # colors = np.zeros((num_colorings, n), dtype=np.uint8)
        # for v in range(n):
        #     if v == 0:
        #         colors[:, v] = 0
        #     else:
        #         colors[:, v] = (indices >> (v-1)) & 1
        
        # This is O(n * 2^(n-1)). For n=14, 14 * 8192 = 114k ops. Very fast.
        
        if n > 16:
            return None, float('inf') # Too large for this approach
            
        indices = np.arange(num_colorings, dtype=np.uint32)
        colors = np.zeros((num_colorings, n), dtype=np.uint8)
        for v in range(1, n):
            colors[:, v] = (indices >> (v - 1)) & 1
        # colors[:, 0] is already 0
        
        for e in edges:
            # Check if colors[:, e] are all equal
            # e is a tuple of 5 vertices
            # ref_col = colors[:, e[0]]
            # others = [colors[:, v] for v in e[1:]]
            # is_mono = np.all(np.stack([ref_col] + others, axis=1) == ref_col[:, None], axis=1)
            
            # Faster:
            # Check if all bits in e are same as e[0]
            # diff = np.any(colors[:, e[1:]] != colors[:, e[0:1]], axis=1)
            # is_mono = ~diff
            
            # Since e has 5 vertices:
            c0 = colors[:, e[0]]
            c1 = colors[:, e[1]]
            c2 = colors[:, e[2]]
            c3 = colors[:, e[3]]
            c4 = colors[:, e[4]]
            
            # is_mono = (c0 == c1) & (c0 == c2) & (c0 == c3) & (c0 == c4)
            is_mono = (c0 == c1) & (c0 == c2) & (c0 == c3) & (c0 == c4)
            
            # Convert boolean array to integer bitmask
            # np.packbits or int.from_bytes
            # is_mono is shape (num_colorings,)
            # We need to pack it into an int.
            # num_colorings might not be a multiple of 8.
            # Pad to multiple of 8?
            # Or just use int.from_bytes(is_mono.tobytes(), 'little')
            
            mask = int.from_bytes(is_mono.tobytes(), 'little')
            masks.append(mask)
            
        universe_mask = (1 << num_colorings) - 1
        
        best_m = float('inf')
        best_edges = None
        
        for iter in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Randomize edge order
            edge_indices = list(range(m_edges))
            random.shuffle(edge_indices)
            
            # Greedy Set Cover
            uncovered = universe_mask
            selected = []
            
            # Precompute coverage for each edge relative to current uncovered?
            # No, greedy picks max overlap with uncovered.
            # We can maintain a list of (coverage, edge_idx)
            # But coverage changes as uncovered changes.
            
            # Simple Greedy:
            # At each step, find edge with max popcount(uncovered & mask_e)
            
            # Optimization:
            # Instead of recalculating popcount for all edges every step,
            # we can just do it. m_edges is small (< 3000).
            
            while uncovered:
                best_edge_idx = -1
                best_cov = 0
                for idx in edge_indices:
                    # Calculate overlap
                    # popcount of (uncovered & masks[idx])
                    # Python int.bit_count() is fast (Python 3.10+)
                    cov = (uncovered & masks[idx]).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_edge_idx = idx
                
                if best_cov == 0:
                    # Cannot cover remaining? Should not happen if universe is covered by union of all masks
                    # But if we ran out of edges? No, we reuse all edges.
                    # If best_cov is 0, it means no edge covers any uncovered coloring.
                    # This implies the hypergraph cannot be made non-2-colorable with these edges?
                    # Actually, the union of ALL masks is the universe?
                    # Yes, for any coloring, there is some 5-subset that is monochromatic?
                    # Not necessarily.
                    # Wait, if we take ALL edges, is the hypergraph non-2-colorable?
                    # Yes, complete hypergraph is non-2-colorable.
                    # So union of all masks must be universe.
                    # So best_cov should be > 0 if uncovered != 0.
                    break
                
                selected.append(best_edge_idx)
                uncovered &= ~masks[best_edge_idx]
                
            # Refinement: Remove redundant edges
            # Try removing each edge, check if uncovered becomes non-zero
            # We need to recompute uncovered from scratch or maintain it carefully.
            # Recomputing is O(m * 2^n) which is fine.
            
            improved = True
            while improved:
                improved = False
                # Iterate in reverse to allow removal
                for i in range(len(selected) - 1, -1, -1):
                    if time.time() > deadline - 2:
                        break
                    # Try removing selected[i]
                    trial_selected = selected[:i] + selected[i+1:]
                    # Check if trial_selected covers universe
                    union_mask = 0
                    for idx in trial_selected:
                        union_mask |= masks[idx]
                    if union_mask == universe_mask:
                        selected = trial_selected
                        improved = True
                        break # Restart loop after removal
            
            if len(selected) < best_m:
                best_m = len(selected)
                # Convert indices to actual edges
                best_edges = [edges[idx] for idx in selected]
                
        return best_edges, best_m

    best_edges_global = None
    best_m_global = float('inf')
    
    # Try different n
    # n=9 is very fast, try many iters
    # n=10, 11 moderate
    # n=12, 13 slower
    
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
            
        # Estimate time per iteration
        # n=9: 126 edges, 256 colorings. Very fast.
        # n=14: 2002 edges, 8192 colorings. Slower.
        
        if n <= 10:
            iters = 50
        elif n <= 12:
            iters = 20
        else:
            iters = 5
            
        edges, m = get_best_hypergraph(n, iters)
        
        if edges is not None and m < best_m_global:
            best_m_global = m
            best_edges_global = edges
            # Update n for output
            n_global = n
            
    if best_edges_global is None:
        # Fallback to complete hypergraph on 9 vertices
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        return {'n': n, 'edges': edges}
        
    # Ensure vertices are 0..n-1
    # The edges are already tuples of vertices 0..n-1
    return {'n': n_global, 'edges': [list(e) for e in best_edges_global]}

