import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency as list of sets for easy intersection, or bitmasks
    # Bitmasks are faster for intersection and subset checks
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    
    def bits_to_list(mask):
        """Convert bitmask to list of set bit positions."""
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            result.append(idx)
            temp ^= lsb
        return result

    def has_k4_in_mask(adj, mask):
        """Check if there's a K4 in the subgraph induced by vertices in mask."""
        if mask == 0:
            return False
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            # For each pair of neighbors of u in mask
            neigh_in_mask = adj[u] & mask & ~(1 << u)
            neigh_list = bits_to_list(neigh_in_mask)
            for j in range(len(neigh_list)):
                v = neigh_list[j]
                if adj[u] & (1 << v):
                    # u and v are connected, check if they have common neighbor w in mask
                    common = adj[u] & adj[v] & mask & ~(1 << u) & ~(1 << v)
                    if common == 0:
                        continue
                    # Check if common contains an edge (w, x)
                    c_list = bits_to_list(common)
                    for a in c_list:
                        if adj[a] & common & ~(1 << a):
                            return True
        return False

    def has_i4_in_mask(adj, mask):
        """Check if there's an independent set of size 4 in the subgraph induced by vertices in mask."""
        if mask == 0:
            return False
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            # Non-neighbors of u in mask (excluding u itself)
            non_neigh_in_mask = mask & ~adj[u] & ~(1 << u)
            non_neigh_list = bits_to_list(non_neigh_in_mask)
            for j in range(len(non_neigh_list)):
                v = non_neigh_list[j]
                # u and v are not connected, check if they have common non-neighbor w in mask
                # w must not be adjacent to u or v
                common_non_neigh = mask & ~adj[u] & ~adj[v] & ~(1 << u) & ~(1 << v)
                if common_non_neigh == 0:
                    continue
                # Check if common_non_neigh contains a non-edge (w, x)
                c_list = bits_to_list(common_non_neigh)
                for a in c_list:
                    for b in c_list:
                        if a == b:
                            continue
                        if not (adj[a] & (1 << b)):
                            return True
        return False

    def can_add_vertex(adj, n, neighbors_mask):
        """Check if we can add a new vertex connected to exactly the vertices in neighbors_mask."""
        # Check that G[neighbors] has no K4
        if has_k4_in_mask(adj, neighbors_mask):
            return False
        # Check that G[non-neighbors] has no I4
        all_mask = (1 << n) - 1
        non_neighbors_mask = all_mask & ~neighbors_mask
        if has_i4_in_mask(adj, non_neighbors_mask):
            return False
        return True

    def is_valid(adj, n):
        """Check if the graph has no K5 and no I5."""
        # Check for K5
        for u in range(n):
            if has_k4_in_mask(adj, adj[u]):
                return False
        # Check for I5
        all_mask = (1 << n) - 1
        for u in range(n):
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_in_mask(adj, non_neigh):
                return False
        return True

    # Verify seed
    if not is_valid(adj_masks, n):
        # Fallback to empty edges if seed is broken (shouldn't happen)
        return n, []

    best_n = n
    best_adj = adj_masks[:]

    # 2. Try to extend using backtracking search
    current_n = n
    current_adj = adj_masks[:]
    
    # Use a priority queue approach: try different neighbor set sizes
    # Start with sizes close to n/2 (balanced degree)
    
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline:
            break
            
        found = False
        
        # Try different neighbor set sizes
        # For a Ramsey graph, degrees should be balanced around n/2
        size_candidates = []
        mid = current_n // 2
        for offset in range(0, current_n // 2 + 1):
            if mid - offset >= 0:
                size_candidates.append(mid - offset)
            if mid + offset <= current_n and mid + offset != mid - offset:
                size_candidates.append(mid + offset)
        
        random.shuffle(size_candidates)
        
        max_attempts_per_size = 1000
        
        for size in size_candidates:
            if time.time() > deadline:
                break
            if found:
                break
            if size < 0 or size > current_n:
                continue
                
            attempts = max_attempts_per_size
            while attempts > 0 and not found:
                if time.time() > deadline:
                    break
                attempts -= 1
                
                # Random neighbor set of given size
                neighbors = random.sample(range(current_n), size)
                neighbors_mask = 0
                for v in neighbors:
                    neighbors_mask |= (1 << v)
                
                if can_add_vertex(current_adj, current_n, neighbors_mask):
                    # Add vertex
                    found = True
                    new_n = current_n + 1
                    new_adj = current_adj + [0]
                    for v in neighbors:
                        new_adj[v] |= (1 << new_n)
                        new_adj[new_n] |= (1 << v)
                    
                    current_n = new_n
                    current_adj = new_adj
                    best_n = current_n
                    best_adj = current_adj[:]
                    break

        if not found:
            # If we can't add this vertex, try a few restarts with different random seeds
            restarts = 5
            for r in range(restarts):
                if time.time() > deadline:
                    break
                # Reset to best state
                current_n = best_n
                current_adj = best_adj[:]
                
                # Try again with different random choices
                for size in size_candidates[:10]:
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    if size < 0 or size > current_n:
                        continue
                        
                    attempts = 500
                    while attempts > 0 and not found:
                        if time.time() > deadline:
                            break
                        attempts -= 1
                        
                        neighbors = random.sample(range(current_n), size)
                        neighbors_mask = 0
                        for v in neighbors:
                            neighbors_mask |= (1 << v)
                        
                        if can_add_vertex(current_adj, current_n, neighbors_mask):
                            found = True
                            new_n = current_n + 1
                            new_adj = current_adj + [0]
                            for v in neighbors:
                                new_adj[v] |= (1 << new_n)
                                new_adj[new_n] |= (1 << v)
                            
                            current_n = new_n
                            current_adj = new_adj
                            best_n = current_n
                            best_adj = current_adj[:]
                            break
                    if found:
                        break
            
            if not found:
                break

    # Reconstruct edges from best_adj
    final_edges = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
                
    return best_n, final_edges
# EVOLVE-BLOCK-END
