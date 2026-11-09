import random
import itertools
import time

# Precompute Q3 properties and subsets
n = 16
all_subsets = list(itertools.combinations(range(n), 8))
num_subsets = len(all_subsets)

# Map global edges to local indices for each subset
# local_edge_map[subset_idx][global_u, global_v] = local_index (0..27)
# Also map local_index to global (u,v)
# And global (u,v) to list of (subset_idx, local_index)

# Precompute Q3 masks for each subset
# q3_masks[subset_idx] = list of 840 28-bit masks
# q3s_by_local_edge[subset_idx][local_l] = list of Q3 masks containing local edge l

# Step 1: Generate all Q3 subgraphs in K8
# Vertices 0..7
# Edges where popcount(i^j)==1
q3_edges_local = []
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_edges_local.append((i, j))

# Map local edges (0..7, 0..7) to local edge index 0..27
local_edge_to_idx = {}
idx = 0
for i in range(8):
    for j in range(i+1, 8):
        local_edge_to_idx[(i, j)] = idx
        idx += 1

# Generate all 8! permutations of 0..7 to find all isomorphic copies
# But we need unique subgraphs.
# A subgraph is defined by the set of 12 edges.
# We can generate all permutations, map them to edge sets, and use a set to deduplicate.
# Permutation p maps Q3 vertex i to subset vertex p[i].
# The edges in the subset are (p[i], p[j]) for (i,j) in q3_edges_local.

all_q3_edge_sets = set()
for p in itertools.permutations(range(8)):
    edges = set()
    for i, j in q3_edges_local:
        u, v = p[i], p[j]
        if u > v:
            u, v = v, u
        edges.add((u, v))
    all_q3_edge_sets.add(frozenset(edges))

# Convert to list of tuples for consistent ordering
q3_edge_list = [list(e) for e in all_q3_edge_sets]
num_q3_in_k8 = len(q3_edge_list) # Should be 840

# Precompute for each subset
# We need:
# 1. subset_masks: array of size num_subsets, 28-bit int
# 2. For each subset, a mapping from global edge (u,v) to local index
# 3. For each subset, q3_masks: list of 28-bit ints
# 4. For each subset, q3s_by_local_edge: list of 28 lists, each containing Q3 masks that include that local edge

# To speed up initialization, we can do this in a loop
# However, 12870 subsets * 840 Q3s * 12 edges is ~130M operations.
# In Python, this might take 5-10 seconds.

# Let's optimize the precomputation structure
# global_edge_to_local_info: dict mapping (u,v) to list of (subset_idx, local_idx)
# This allows fast lookup when flipping an edge.

# We will store:
# subset_data = [None] * num_subsets
# Each subset_data[i] will be a dict or array holding:
# - local_edge_indices: array of size 28, mapping local_idx -> global (u,v) ? No, we need global -> local.
# - q3_masks: list of 840 ints
# - q3s_by_local_edge: list of 28 lists of ints (the Q3 masks)

# Let's build the mapping for each subset
# local_edge_to_global[subset_idx][local_idx] = (u, v)
# global_edge_to_local[subset_idx][(u, v)] = local_idx

print("Precomputing subsets...", flush=True)
start_time = time.time()

subset_local_edges = [None] * num_subsets
subset_q3_masks = [None] * num_subsets
subset_q3s_by_local_edge = [None] * num_subsets
global_edge_to_subset_local = {} # (u,v) -> list of (subset_idx, local_idx)

for si, subset in enumerate(all_subsets):
    # Map local indices 0..7 to global vertices
    # local_vertex[0..7] = subset[0..7]
    # We need to map the 28 edges of K8 to local indices 0..27
    # local_edge_idx = 0
    # local_edges = []
    # for i in range(8):
    #   for j in range(i+1, 8):
    #     u, v = subset[i], subset[j]
    #     local_edges.append((u, v))
    #     global_edge_to_subset_local[(u, v)].append((si, local_edge_idx))
    #     local_edge_idx += 1
    
    local_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            u, v = subset[i], subset[j]
            if u > v: u, v = v, u
            local_edges.append((u, v))
            
    # Initialize structures
    q3_masks = []
    q3s_by_local_edge = [[] for _ in range(28)]
    
    for q3_edges in q3_edge_list:
        # q3_edges is a list of 12 (local_i, local_j) pairs in 0..7
        # Map these to local_edge_indices in this subset
        mask = 0
        for le in q3_edges:
            li, lj = le
            # Find the index of (subset[li], subset[lj]) in local_edges
            # We can precompute a lookup for (li, lj) -> local_edge_idx
            # Since li, lj are 0..7, we can use a 8x8 matrix or just linear search (slow)
            # Let's create a lookup table for this subset
            # Actually, we can compute the index directly:
            # The edge (li, lj) corresponds to the pair (subset[li], subset[lj]).
            # In the loop above, local_edges[k] corresponds to (subset[i], subset[j])
            # where i, j are the indices in the range(8) loop.
            # So the local edge index is determined by i, j.
            # We can compute the index:
            # i = li, j = lj (assuming li < lj? No, q3_edges has i<j)
            # Wait, q3_edges_local was generated with i<j.
            # So li < lj.
            # The index in the 28-edge list is:
            # count of pairs (a,b) with a < b such that (a,b) < (li, lj) lexicographically?
            # No, the order is:
            # (0,1), (0,2), ..., (0,7)
            # (1,2), ..., (1,7)
            # ...
            # (6,7)
            # Index = sum_{k=0}^{li-1} (7-k) + (lj - li - 1)
            # Let's verify:
            # li=0, lj=1: sum=0, +0 = 0. Correct.
            # li=0, lj=7: sum=0, +6 = 6. Correct.
            # li=1, lj=2: sum_{k=0}^{0}(7-k)=7. +0 = 7. Correct.
            
            le_idx = 0
            for k in range(li):
                le_idx += (7 - k)
            le_idx += (lj - li - 1)
            
            mask |= (1 << le_idx)
            q3s_by_local_edge[le_idx].append(mask)
            
        q3_masks.append(mask)
        
    subset_local_edges[si] = local_edges
    subset_q3_masks[si] = q3_masks
    subset_q3s_by_local_edge[si] = q3s_by_local_edge
    
    # Update global mapping
    for le_idx, (u, v) in enumerate(local_edges):
        if (u, v) not in global_edge_to_subset_local:
            global_edge_to_subset_local[(u, v)] = []
        global_edge_to_subset_local[(u, v)].append((si, le_idx))

print(f"Precomputation done in {time.time() - start_time:.2f}s", flush=True)

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    # Initialize random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Initialize subset_masks
    # subset_masks[si] is a 28-bit integer
    subset_masks = [0] * num_subsets
    for si in range(num_subsets):
        mask = 0
        for le_idx, (u, v) in enumerate(subset_local_edges[si]):
            if (adj_bits[u] >> v) & 1:
                mask |= (1 << le_idx)
        subset_masks[si] = mask
    
    # Calculate initial energy (number of violated clauses)
    energy = 0
    for si in range(num_subsets):
        m = subset_masks[si]
        for q3_mask in subset_q3_masks[si]:
            if (m & q3_mask) == q3_mask:
                energy += 1
                
    best_energy = energy
    best_adj_bits = adj_bits[:]
    best_subset_masks = subset_masks[:]
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute list of all global edges for random selection
    all_global_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_global_edges.append((i, j))
            
    current_energy = energy
    
    while time.time() - start_time < time_limit - 5:
        # Pick random edge
        u, v = random.choice(all_global_edges)
        is_present = (adj_bits[u] >> v) & 1
        
        # Calculate Delta Energy
        delta_e = 0
        # Get all (si, le_idx) for this edge
        infos = global_edge_to_subset_local.get((u, v), [])
        
        if is_present:
            # Removing edge
            # A Q3 is "lost" if it contains this edge and all other 11 edges are present
            for si, le_idx in infos:
                m = subset_masks[si]
                # Check Q3s in this subset that contain le_idx
                for q3_mask in subset_q3s_by_local_edge[si][le_idx]:
                    # q3_mask contains le_idx bit
                    # We need to check if (m & q3_mask) == q3_mask
                    # But we are removing the edge, so we check if it WAS complete
                    # If (m & q3_mask) == q3_mask, then it is currently complete.
                    # Removing the edge makes it incomplete.
                    # So delta_e -= 1 for each such Q3.
                    if (m & q3_mask) == q3_mask:
                        delta_e -= 1
        else:
            # Adding edge
            # A Q3 is "gained" if it contains this edge and all other 11 edges are present
            for si, le_idx in infos:
                m = subset_masks[si]
                for q3_mask in subset_q3s_by_local_edge[si][le_idx]:
                    # Check if other 11 edges are present
                    # q3_mask has bit le_idx set.
                    # other_mask = q3_mask ^ (1 << le_idx)
                    # If (m & other_mask) == other_mask, then adding edge completes it.
                    other_mask = q3_mask ^ (1 << le_idx)
                    if (m & other_mask) == other_mask:
                        delta_e += 1
                        
        new_energy = current_energy + delta_e
        
        # SA Acceptance
        if new_energy < current_energy:
            accept = True
        else:
            if T > 0:
                delta = new_energy - current_energy
                # P = exp(-delta / T)
                if random.random() < pow(2.718281828, -delta / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
                
        if accept:
            # Apply flip
            if is_present:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
            # Update subset masks
            for si, le_idx in infos:
                if is_present:
                    subset_masks[si] &= ~(1 << le_idx)
                else:
                    subset_masks[si] |= (1 << le_idx)
                    
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj_bits = adj_bits[:]
                best_subset_masks = subset_masks[:]
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Use best solution
    adj_bits = best_adj_bits
    
    # Final cleanup: remove any remaining Q3s if best_energy > 0
    # This should be rare if SA converged well
    if best_energy > 0:
        # Fallback to greedy removal
        while time.time() - start_time < time_limit - 0.5:
            # Find a Q3
            found = False
            for si in range(num_subsets):
                m = best_subset_masks[si]
                for q3_mask in subset_q3_masks[si]:
                    if (m & q3_mask) == q3_mask:
                        # Found one
                        # Remove a random edge from this Q3
                        # Get the edges of this Q3
                        edges_in_q3 = []
                        for le_idx in range(28):
                            if (q3_mask >> le_idx) & 1:
                                u, v = subset_local_edges[si][le_idx]
                                edges_in_q3.append((u, v))
                        
                        u, v = random.choice(edges_in_q3)
                        best_adj_bits[u] &= ~(1 << v)
                        best_adj_bits[v] &= ~(1 << u)
                        
                        # Update masks
                        infos = global_edge_to_subset_local.get((u, v), [])
                        for s_idx, l_idx in infos:
                            best_subset_masks[s_idx] &= ~(1 << l_idx)
                            
                        found = True
                        break
                if found:
                    break
            if not found:
                break
                
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
