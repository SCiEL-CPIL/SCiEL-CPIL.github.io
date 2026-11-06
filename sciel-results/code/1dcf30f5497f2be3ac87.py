import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    # Precompute all edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)  # 120
    edge_index = {e: i for i, e in enumerate(all_edges)}

    # Precompute Q3 structure
    # Q3 vertices are 0..7. Edges connect i, j if popcount(i^j)==1.
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
    
    # There are 12 edges in Q3
    # We need to find all embeddings of Q3 into K8 (8 vertices).
    # An embedding is a permutation of 8 vertices.
    # Two embeddings are equivalent if they induce the same set of edges in K8.
    # We represent a set of edges in K8 as a bitmask of length 28 (since K8 has 28 edges).
    
    k8_edges = []
    local_edge_to_bit = {}
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    q3_masks = set()
    # Generate all 8! = 40320 permutations
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    # There are 1344 distinct masks.

    # Precompute for each 8-subset of vertices, the mapping from local edge bits to global edge indices
    subsets = list(combinations(range(n), 8))
    # subset_edge_maps[subset] = list of (global_edge_index, local_edge_bit)
    subset_edge_maps = {}
    for subset in subsets:
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                mapping.append((ge_idx, le_bit))
        subset_edge_maps[subset] = mapping
        
    # For fast checking:
    # We want to check if a given global mask (120 bits) contains a Q3.
    # A Q3 is contained if there exists a subset S of 8 vertices such that the projection of the global mask onto S is a superset of one of the q3_masks.
    
    # Optimization:
    # Instead of checking all subsets for every edge addition, we can precompute for each global edge, which subsets it belongs to.
    # But that's not quite right.
    # Let's think about the cost.
    # Checking if a mask is Q3-free:
    # Iterate over 128 subsets.
    # For each subset, extract the 28-bit local mask.
    # Check if local_mask contains any of the 1344 q3_masks.
    # This is expensive if done naively.
    
    # Better approach for local search:
    # When we add an edge (u,v), we only need to check subsets that contain both u and v.
    # How many such subsets?
    # Total subsets: C(16,8) = 12870.
    # Subsets containing u and v: C(14, 6) = 3003.
    # So for each edge addition, we check 3003 subsets.
    # For each subset, we need to check if the local mask contains a Q3.
    # The local mask is 28 bits. We have 1344 Q3 masks.
    # We can precompute for each 28-bit mask, the list of Q3 masks it contains? No, 2^28 is too large.
    
    # Alternative:
    # For a fixed subset, the local mask is determined by the 28 edges in that subset.
    # We can precompute for each subset, and for each possible local mask, whether it contains a Q3.
    # But local mask has 28 bits, so 2^28 states. Too many.
    
    # Let's stick to the direct check but optimize the inner loop.
    # For a given subset and local mask, we check:
    # for qm in q3_masks_list:
    #     if (local_mask & qm) == qm: return True
    # This is 1344 iterations per subset.
    # 3003 subsets * 1344 checks = ~4 million operations per edge addition.
    # In Python, this might be slow. 4M ops ~ 0.1-0.5s? Maybe too slow for many iterations.
    
    # We need a faster way to check if local_mask contains any q3_mask.
    # We can use a lookup table for smaller chunks?
    # Or, we can precompute for each subset, a bitset of "which Q3 masks are present"?
    # No, that's what we are trying to compute.
    
    # Let's try to optimize the check function using bit operations and precomputed tables.
    # Split the 28-bit local mask into 4 parts of 7 bits? Or 2 parts of 14 bits?
    # If we split into 2 parts of 14 bits, we have 2^14 = 16384 entries per part.
    # For each part, we can precompute which Q3 masks have their bits in that part.
    # This doesn't directly help with the "subset" check.
    
    # Let's try a different strategy:
    # Precompute for each of the 128 subsets, a list of the 1344 Q3 masks projected onto that subset?
    # No, the Q3 masks are already defined on the 8 vertices of the subset.
    
    # How about this:
    # For each subset, we have a 28-bit local mask.
    # We want to know if any of the 1344 Q3 masks is a subset of the local mask.
    # We can precompute for each 28-bit mask... no.
    
    # Let's try to use a SAT solver approach or just optimize the Python loop.
    # Maybe use numpy?
    # q3_masks_list is a list of 1344 integers.
    # We can convert it to a numpy array.
    # local_mask is an integer.
    # We can check: np.any((local_mask & q3_masks_arr) == q3_masks_arr)
    # This is vectorized and might be faster.
    
    import numpy as np
    q3_masks_arr = np.array(q3_masks_list, dtype=np.uint32)
    
    def is_q3_free_in_subset(local_mask, subset_idx):
        # local_mask is a 28-bit integer
        # We check if local_mask contains any Q3 mask.
        # Since q3_masks_arr is the same for all subsets (just interpreted differently),
        # we can use the same array.
        return np.any((local_mask & q3_masks_arr) == q3_masks_arr)
        
    # Wait, the local_mask is formed by the edges in the subset.
    # The Q3 masks are defined on the 8 vertices of the subset.
    # So yes, the check is the same for all subsets.
    
    # Now, for a global mask, how do we check if it's Q3-free?
    # We need to iterate over all subsets that contain the added edge?
    # No, if we are checking the whole graph, we need all subsets.
    # But in local search, we only add one edge at a time.
    # So we only need to check subsets containing that edge.
    
    # Let's precompute for each global edge index, the list of subset indices that contain it.
    # subset_indices is just the index in the `subsets` list.
    # edge_to_subsets = {edge_idx: [subset_idx1, subset_idx2, ...]}
    
    # But we also need the mapping for each subset to extract the local mask.
    # subset_edge_maps[subset] gives the mapping.
    # We need to extract the 28-bit local mask from the global 120-bit mask.
    # This requires iterating over the 28 edges in the subset and checking the global mask.
    # That's 28 bit operations.
    # 3003 subsets * 28 ops = 84000 ops per edge addition.
    # Plus the numpy check: 3003 * (1344 vectorized ops).
    # The numpy check might be the bottleneck.
    
    # Can we precompute the local mask for each subset more efficiently?
    # We can maintain the local masks incrementally?
    # When we add an edge (u,v), the local mask for a subset S containing (u,v) changes by setting one bit.
    # So we can update the local mask in O(1).
    # But we need to store the local mask for each subset.
    # 12870 subsets * 1 integer = 12870 integers. This is fine.
    
    # Let's try this approach:
    # 1. Initialize global_mask = 0.
    # 2. Initialize local_masks = [0] * len(subsets).
    # 3. When adding edge (u,v) with global index ge_idx:
    #    - For each subset S containing (u,v):
    #      - Find the local bit index for (u,v) in S.
    #      - Set that bit in local_masks[S].
    #      - Check if local_masks[S] contains a Q3.
    #    - If any subset now contains a Q3, the addition is invalid.
    # 4. When removing edge (u,v):
    #    - For each subset S containing (u,v):
    #      - Clear the bit in local_masks[S].
    #      - (No need to check for Q3, since removing edges can't create a Q3).
    
    # This is much faster!
    # We need:
    # - edge_to_subsets: dict mapping global edge index to list of (subset_idx, local_bit)
    
    edge_to_subsets = {i: [] for i in range(num_edges)}
    for s_idx, subset in enumerate(subsets):
        mapping = subset_edge_maps[subset]
        for ge_idx, le_bit in mapping:
            edge_to_subsets[ge_idx].append((s_idx, le_bit))
            
    # Now, the check for adding an edge ge_idx:
    # For each (s_idx, le_bit) in edge_to_subsets[ge_idx]:
    #     new_local_mask = local_masks[s_idx] | (1 << le_bit)
    #     if is_q3_free_in_subset(new_local_mask):
    #         continue
    #     else:
    #         return False
    # return True
    
    # And the update:
    # For each (s_idx, le_bit) in edge_to_subsets[ge_idx]:
    #     local_masks[s_idx] |= (1 << le_bit)
    
    # Let's implement this.
    
    def can_add_edge(ge_idx, local_masks):
        for s_idx, le_bit in edge_to_subsets[ge_idx]:
            new_lm = local_masks[s_idx] | (1 << le_bit)
            if not is_q3_free_in_subset(new_lm):
                return False
        return True
        
    def add_edge(ge_idx, local_masks):
        for s_idx, le_bit in edge_to_subsets[ge_idx]:
            local_masks[s_idx] |= (1 << le_bit)
            
    def remove_edge(ge_idx, local_masks):
        for s_idx, le_bit in edge_to_subsets[ge_idx]:
            local_masks[s_idx] &= ~(1 << le_bit)
            
    # Initial solution: Start with a random graph and repair.
    random.seed(42)
    
    best_mask = 0
    best_local_masks = [0] * len(subsets)
    best_count = 0
    
    # Try to find a good initial solution quickly
    # Start with a random graph with p=0.5
    current_mask = 0
    current_local_masks = [0] * len(subsets)
    for i in range(num_edges):
        if random.random() < 0.5:
            current_mask |= (1 << i)
            add_edge(i, current_local_masks)
            
    # Repair: Remove edges that cause Q3
    # We need to find which edges are in Q3s.
    # We can iterate over all subsets and find Q3s.
    # For each Q3 found, remove a random edge from it.
    
    def find_violating_q3_global(mask, local_masks):
        # Find any Q3 in the graph
        for s_idx in range(len(subsets)):
            lm = local_masks[s_idx]
            # Check if lm contains a Q3
            # We need to return the global edges involved.
            # This is harder with the new approach.
            # Let's just return the subset index for now, and handle removal differently.
            # Or, we can just check if it's Q3-free, and if not, try to fix it.
            pass
        return None
        
    # Simpler repair:
    # Just keep removing random edges until Q3-free.
    # But we need to know if it's Q3-free.
    
    def is_q3_free_global(local_masks):
        for lm in local_masks:
            if not is_q3_free_in_subset(lm):
                return False
        return True
        
    # Repair loop
    for _ in range(1000):
        if is_q3_free_global(current_local_masks):
            break
        # Find a violating edge
        # We can just pick a random edge and remove it? No, that's inefficient.
        # Let's find a subset that has a Q3, and remove a random edge from that Q3.
        # To do this, we need to find which Q3 is present in that subset.
        # We can use the q3_masks_list.
        
        found = False
        for s_idx in range(len(subsets)):
            lm = current_local_masks[s_idx]
            if is_q3_free_in_subset(lm):
                continue
            # Find a Q3 mask that is a subset of lm
            # We can iterate through q3_masks_list
            for qm in q3_masks_list:
                if (lm & qm) == qm:
                    # Found a Q3 in this subset
                    # Pick a random edge from this Q3
                    # The Q3 has 12 edges. We need to map the local bits to global edges.
                    # We have the subset, and the Q3 mask.
                    # We can iterate through the 12 bits of qm, and for each set bit,
                    # find the corresponding global edge index.
                    # Then pick one at random and remove it.
                    
                    subset = subsets[s_idx]
                    # We need the mapping from local bit to global edge index for this subset.
                    # We have subset_edge_maps[subset], which is a list of (ge_idx, le_bit).
                    # We can create a dict for this subset: le_bit -> ge_idx
                    # But doing this every time is slow.
                    # Let's precompute this.
                    
                    break
            if found:
                break
                
    # This repair is getting complicated. Let's simplify.
    # Just start with an empty graph and add edges greedily.
    
    current_mask = 0
    current_local_masks = [0] * len(subsets)
    
    # Greedy add
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    for ge_idx in edge_order:
        if can_add_edge(ge_idx, current_local_masks):
            add_edge(ge_idx, current_local_masks)
            current_mask |= (1 << ge_idx)
            
    best_mask = current_mask
    best_local_masks = current_local_masks[:]
    best_count = bin(current_mask).count('1')
    
    # Local search: Try to add more edges, or swap edges.
    # Since we started with a greedy add, we might be at a local optimum where no single edge can be added.
    # We can try to remove an edge and then add more.
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        # Start from the current best
        current_mask = best_mask
        current_local_masks = best_local_masks[:]
        
        # Remove a random edge
        if bin(current_mask).count('1') > 0:
            # Pick a random edge from the current graph
            current_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
            rem_idx = random.choice(current_edges)
            remove_edge(rem_idx, current_local_masks)
            current_mask &= ~(1 << rem_idx)
            
        # Now try to add edges greedily
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing_edges)
        for ge_idx in missing_edges:
            if can_add_edge(ge_idx, current_local_masks):
                add_edge(ge_idx, current_local_masks)
                current_mask |= (1 << ge_idx)
                
        # Check if we improved
        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask
            best_local_masks = current_local_masks[:]
            
    # Final polish: Try to add any remaining edges
    missing_edges = [i for i in range(num_edges) if not (best_mask >> i) & 1]
    random.shuffle(missing_edges)
    for ge_idx in missing_edges:
        if can_add_edge(ge_idx, best_local_masks):
            add_edge(ge_idx, best_local_masks)
            best_mask |= (1 << ge_idx)
            best_count += 1
            
    # Convert best_mask to list of edges
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
