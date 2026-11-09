import random
import itertools
import time

# Precompute Q3 properties
q3_edges_local = [(i, j) for i in range(8) for j in range(i+1, 8) if bin(i ^ j).count('1') == 1]

# Precompute all 8-vertex subsets and their corresponding global edge indices for the identity Q3 mapping
n = 16
all_subsets = list(itertools.combinations(range(n), 8))
subset_edge_indices = []
for subset in all_subsets:
    # subset is a tuple of 8 vertices
    # We map local index k to subset[k]
    # The edges of Q3 are (i,j) where popcount(i^j)==1
    # We need the global edge index for (subset[i], subset[j])
    # Global edge index: we can just use a set or a 2D array, or a single integer bitmask
    # Let's use a flat list of edge indices for fast lookup
    # Edge (u,v) with u<v. Index = u*15 - u*(u+1)//2 + (v-u-1)? 
    # Simpler: just store the pairs (u,v) and check adj_bits[u] & (1<<v)
    # But for speed, let's precompute a bitmask of the 12 edges?
    # No, 12 edges don't fit in a small int easily if we want to check "all present".
    # We can precompute for each subset a list of (u, v) pairs.
    edges_in_subset = []
    for i, j in q3_edges_local:
        u = subset[i]
        v = subset[j]
        if u > v:
            u, v = v, u
        edges_in_subset.append((u, v))
    subset_edge_indices.append(edges_in_subset)

# Precompute which subsets contain a specific global edge (for incremental updates if needed, or just for analysis)
# Actually, for the SA cost, we just evaluate the whole thing.
# To speed up SA, we can only evaluate subsets affected by the move?
# If we flip edge (u,v), only subsets containing u and v are affected.
# Let's precompute `subsets_by_edge`
subsets_by_edge = {e: [] for e in [(i, j) for i in range(n) for j in range(i+1, n)]}
# Map edge tuple to index?
# Let's just use a dict of sets.
# For each subset index, for each edge in its Q3 mapping, add subset index to subsets_by_edge[edge]
for idx, subset in enumerate(all_subsets):
    for edge in subset_edge_indices[idx]:
        subsets_by_edge[edge].append(idx)

def count_q3_identity(adj_bits):
    """Count subsets that contain the identity Q3 embedding."""
    count = 0
    for idx, edges in enumerate(subset_edge_indices):
        # Check if all edges are present
        is_q3 = True
        for u, v in edges:
            if not (adj_bits[u] >> v) & 1:
                is_q3 = False
                break
        if is_q3:
            count += 1
    return count

def construct():
    n = 16
    total_edges = n * (n-1) // 2
    start_time = time.time()
    time_limit = 115
    
    # Initialize random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
                adj_bits[i] |= (1 << j)
                adj_bits[v] |= (1 << i) # Wait, v is not defined in this scope? 
                # Fix:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    current_q3_count = count_q3_identity(adj_bits)
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    best_adj_bits = adj_bits[:]
    best_score = 100 * current_q3_count - current_edges
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of all edges for random selection
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while T > min_T:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Determine if we are adding or removing
        is_present = (adj_bits[u] >> v) & 1
        
        # Calculate delta in q3_count and edges
        delta_edges = -1 if is_present else 1
        
        # Update q3 count incrementally?
        # If we flip (u,v), only subsets containing (u,v) in their identity mapping are affected.
        # But wait, (u,v) might not be in the identity mapping of a subset even if the subset contains u and v.
        # We precomputed `subsets_by_edge` for the *identity* mapping edges.
        # So we only need to check those subsets.
        
        affected_subsets = subsets_by_edge.get((u, v), [])
        
        delta_q3 = 0
        for idx in affected_subsets:
            edges = subset_edge_indices[idx]
            # Check if this subset was a Q3 before
            # We need to know the state of the other 11 edges
            # This is slow if we check all 11.
            # Let's just re-evaluate the whole subset?
            # Or, store a bitmask of which of the 12 edges are present?
            # Let's optimize by storing for each subset a 12-bit mask of present edges.
            # But that requires updating 12 bits per subset per move.
            
            # Given the small number of affected subsets (approx 3000), and 12 checks each:
            # 3000 * 12 = 36000 checks.
            # This might be okay.
            
            # Let's just do a full check for the affected subsets.
            # We need to know if it was Q3 before and if it is Q3 after.
            
            # To do this efficiently, we can just count how many of the 12 edges are present.
            # If count == 12, it's a Q3.
            
            # Let's refactor:
            # We will just re-calculate the q3 status of the affected subsets.
            # But we need the "before" status.
            # We can store `subset_is_q3` array.
            
            pass # Placeholder for logic
            
        # Actually, incremental update is tricky. Let's just recalculate `current_q3_count` for the whole graph?
        # 12870 subsets * 12 checks = 154k checks.
        # In Python, 154k simple checks might take ~0.01-0.05s.
        # If we do 1000 SA steps, that's 10-50s. Feasible.
        
        # Let's try this simpler approach:
        # 1. Flip edge
        # 2. Recalculate total q3 count
        # 3. Calculate new score
        # 4. Accept or reject
        
        # Flip edge
        if is_present:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        else:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        new_q3_count = count_q3_identity(adj_bits)
        new_edges = current_edges + delta_edges
        new_score = 100 * new_q3_count - new_edges
        
        delta_score = new_score - best_score # Or current score?
        # SA usually compares to current state, not best.
        # Let's keep track of current state score.
        current_score = 100 * current_q3_count - current_edges
        delta_score = new_score - current_score
        
        if delta_score < 0 or random.random() < pow(2.718, -delta_score / T):
            # Accept
            current_q3_count = new_q3_count
            current_edges = new_edges
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_adj_bits = adj_bits[:]
        else:
            # Reject: revert
            if is_present:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            else:
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
        T *= cooling

    # Final cleanup: Remove any remaining Q3s (including permuted ones)
    adj_bits = best_adj_bits
    
    # Use the original robust check to clean up
    # Re-define contains_q3_fast and find_q3_subset here or use globals?
    # We can just use a simple loop with the backtracking checker from the original code.
    
    # Let's copy the necessary helpers from the original code to ensure correctness
    def contains_q3_fast(subset, adj_bits):
        local_adj = [0]*8
        for i in range(8):
            vi = subset[i]
            for j in range(8):
                if i != j and (adj_bits[vi] >> subset[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False
        
        total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
        if total_deg < 24:
            return False
        
        mapping = [-1]*8
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            used_mask = 0
            for u in range(q3_v):
                used_mask |= (1 << mapping[u])
            
            for s_idx in range(8):
                if (used_mask >> s_idx) & 1:
                    continue
                valid = True
                for u in range(q3_v):
                    if (q3_adj[u] >> q3_v) & 1:
                        if not ((local_adj[mapping[u]] >> s_idx) & 1):
                            valid = False
                            break
                if valid:
                    mapping[q3_v] = s_idx
                    if backtrack(q3_v + 1):
                        return True
                    mapping[q3_v] = -1
            return False
        
        for first in range(8):
            mapping[0] = first
            if backtrack(1):
                return True
            mapping[0] = -1
        return False

    def find_q3_subset(adj_bits, n):
        for _ in range(50):
            subset = random.sample(range(n), 8)
            if contains_q3_fast(subset, adj_bits):
                return subset
        for subset in itertools.combinations(range(n), 8):
            if contains_q3_fast(subset, adj_bits):
                return list(subset)
        return None

    while time.time() - start_time < time_limit - 1:
        q3_sub = find_q3_subset(adj_bits, n)
        if q3_sub is None:
            break
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                    edges_in_q3.append((q3_sub[i], q3_sub[j]))
        if not edges_in_q3:
            break
        u, v = random.choice(edges_in_q3)
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
