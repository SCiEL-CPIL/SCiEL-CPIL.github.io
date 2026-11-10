import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 16
    random.seed(42)
    
    # Precompute all 8-subsets of vertices
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # Generate Q3 edge patterns in K8
    # Base Q3 edges: vertices 0-7, edge if hamming distance is 1
    base_edges = frozenset(
        tuple(sorted((i, j))) 
        for i in range(8) 
        for j in range(i+1, 8) 
        if bin(i ^ j).count('1') == 1
    )
    
    # Generate all unique Q3 edge sets (840 patterns)
    unique_patterns = set()
    for perm in itertools.permutations(range(8)):
        new_edges = frozenset(
            tuple(sorted((perm[i], perm[j]))) 
            for i, j in base_edges
        )
        unique_patterns.add(new_edges)
    
    q3_patterns = list(unique_patterns)
    num_patterns = len(q3_patterns)
    
    # Precompute: for each subset index, store the subset
    # Precompute: for each edge (u,v), which subset indices contain it
    # Edge (u,v) is in subset S if u,v in S
    # C(14,6) = 3003 subsets contain any given edge
    
    # Build edge_to_subset_indices mapping
    # We'll use a dictionary: edge -> list of subset indices
    # But storing 120 edges * 3003 indices = 360k entries, manageable
    
    # Actually, let's use a more efficient approach:
    # For each subset, precompute its "signature" as a bitmask of edges in K8
    # Then checking if it contains a Q3 pattern is: pattern_mask & subset_mask == pattern_mask
    
    # Let's use bitmasks for edges in K8: 28 edges, so 28-bit masks
    # Edge (i,j) in 0..7 maps to bit position i*8 + j (or some unique index)
    
    # Create edge-to-bit mapping for K8
    edge_bit = {}
    bit_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            edge_bit[(i,j)] = 1 << bit_idx
            bit_idx += 1
    
    # Convert Q3 patterns to bitmasks
    pattern_masks = []
    for pat in q3_patterns:
        mask = 0
        for e in pat:
            mask |= edge_bit[e]
        pattern_masks.append(mask)
    
    # For each subset, we need to compute its edge bitmask given the current graph
    # This is expensive to do from scratch each time
    
    # Alternative approach: use a smarter search strategy
    # Start from a known good initial graph and try to add edges
    
    # Strategy: 
    # 1. Start with a Q3-free graph (e.g., complete bipartite K8,8 or similar)
    # 2. Try to add edges one by one, checking if it creates a Q3
    # 3. If adding an edge creates a Q3, skip it or try another
    
    # Let's start with K8,8: vertices 0-7 and 8-15, all cross edges, no within-group edges
    # K8,8 has 64 edges and is Q3-free? Let's verify.
    # Q3 is 3-regular bipartite. In K8,8, the induced subgraph on any 8 vertices
    # could contain a Q3.
    # Actually, K8,8 contains many Q3 subgraphs. So this won't work.
    
    # Better: start with a random sparse graph and add edges greedily
    
    # Or: use the greedy removal approach but with better heuristics
    
    # Let's use a different approach:
    # 1. Start with K16
    # 2. Use local search: find a Q3, remove an edge, repeat
    # 3. But with better edge selection and time management
    
    # Precompute subset edge bitmasks for initial K16
    # In K16, every subset has all 28 edges present
    full_mask = (1 << 28) - 1
    
    # For current graph, maintain:
    # - set of edges
    # - for each subset, whether it's "bad" (contains a Q3)
    # - for each bad subset, which pattern is violated
    
    # Let's implement an efficient local search:
    
    # State: set of edges in the graph
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
    
    # Helper: compute edge bitmask for a subset given current edges
    def get_subset_mask(S):
        mask = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                e = (min(u,v), max(u,v))
                if e in edges:
                    mask |= edge_bit[(i,j)]
        return mask
    
    # Helper: check if a mask contains any Q3 pattern
    def contains_q3_mask(mask):
        for pm in pattern_masks:
            if (mask & pm) == pm:
                return True
        return False
    
    # Precompute for each subset, the list of (pattern_mask, edge_indices_in_pattern)
    # to allow quick identification of which edge to remove
    
    # Let's use a greedy approach with randomization:
    # 1. Find all bad subsets
    # 2. Pick a bad subset
    # 3. Find a Q3 pattern in it
    # 4. Remove a random edge from that pattern
    # 5. Repeat until no bad subsets or time out
    
    # To speed up, maintain a list of bad subsets and update incrementally
    
    # Initial: all subsets are bad (K16 contains Q3 in every 8-subset)
    bad_subset_indices = list(range(num_subsets))
    
    # Precompute: for each edge, which subsets contain it
    # This allows us to only check subsets that might have changed
    edge_to_subsets = {}
    for idx, S in enumerate(all_subsets):
        S_set = set(S)
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                e = (min(u,v), max(u,v))
                if e not in edge_to_subsets:
                    edge_to_subsets[e] = set()
                edge_to_subsets[e].add(idx)
    
    # Convert to lists for faster iteration
    for e in edge_to_subsets:
        edge_to_subsets[e] = list(edge_to_subsets[e])
    
    # Main loop
    while bad_subset_indices and time.time() < deadline:
        # Pick a random bad subset
        idx = random.randint(0, len(bad_subset_indices) - 1)
        S = all_subsets[bad_subset_indices[idx]]
        
        # Verify it's still bad
        mask = get_subset_mask(S)
        if not contains_q3_mask(mask):
            # It's no longer bad, remove from list
            bad_subset_indices.pop(idx)
            continue
        
        # Find which pattern is violated
        violated_pattern = None
        for pm in pattern_masks:
            if (mask & pm) == pm:
                violated_pattern = pm
                break
        
        if violated_pattern is None:
            bad_subset_indices.pop(idx)
            continue
        
        # Find edges in the violated pattern that are in the current graph
        # We need to map pattern bits back to edges in S
        pattern_edges_local = []
        for i in range(8):
            for j in range(i+1, 8):
                if violated_pattern & edge_bit[(i,j)]:
                    u, v = S[i], S[j]
                    pattern_edges_local.append((min(u,v), max(u,v)))
        
        # Remove a random edge from the pattern
        e_to_remove = random.choice(pattern_edges_local)
        edges.discard(e_to_remove)
        
        # Update bad_subset_indices:
        # Only subsets containing e_to_remove might have changed
        # For those subsets, recheck if they're still bad
        affected_subsets = edge_to_subsets.get(e_to_remove, [])
        
        new_bad_indices = []
        for sidx in bad_subset_indices:
            if sidx in affected_subsets:
                # Recheck
                S2 = all_subsets[sidx]
                mask2 = get_subset_mask(S2)
                if contains_q3_mask(mask2):
                    new_bad_indices.append(sidx)
                # else: it became good, don't add
            else:
                # Unchanged, still bad
                new_bad_indices.append(sidx)
        
        bad_subset_indices = new_bad_indices
    
    # If we ran out of time, return the current graph
    return [[u, v] for u, v in edges]
