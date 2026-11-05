import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
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

    def count_bits(mask):
        return bin(mask).count('1')

    def has_k4_fast(adj, mask):
        """Check if there's a K4 in the subgraph induced by vertices in mask using bit tricks."""
        verts = bits_to_list(mask)
        nverts = len(verts)
        if nverts < 4:
            return False
        # Try all triples, check if they form K3 and have common neighbor
        for i in range(nverts):
            u = verts[i]
            for j in range(i+1, nverts):
                v = verts[j]
                if not (adj[u] & (1 << v)):
                    continue
                common = adj[u] & adj[v] & mask & ~(1 << u) & ~(1 << v)
                if count_bits(common) < 2:
                    continue
                # Check if any two in common are connected
                cl = bits_to_list(common)
                for a in cl:
                    if adj[a] & common & ~(1 << a):
                        return True
        return False

    def has_i4_fast(adj, mask):
        """Check if there's an independent set of size 4 in the subgraph induced by vertices in mask."""
        verts = bits_to_list(mask)
        nverts = len(verts)
        if nverts < 4:
            return False
        for i in range(nverts):
            u = verts[i]
            non_neigh_u = mask & ~adj[u] & ~(1 << u)
            for j in range(i+1, nverts):
                v = verts[j]
                if adj[u] & (1 << v):
                    continue
                common_non_neigh = non_neigh_u & ~adj[v] & ~(1 << u) & ~(1 << v)
                if count_bits(common_non_neigh) < 2:
                    continue
                # Check if any two in common_non_neigh are non-adjacent
                cnl = bits_to_list(common_non_neigh)
                for a in cnl:
                    for b in cnl:
                        if a == b:
                            continue
                        if not (adj[a] & (1 << b)):
                            return True
        return False

    def can_add_vertex(adj, n, neighbors_mask):
        """Check if we can add a new vertex connected to exactly the vertices in neighbors_mask."""
        # Check that G[neighbors] has no K4
        if has_k4_fast(adj, neighbors_mask):
            return False
        # Check that G[non-neighbors] has no I4
        all_mask = (1 << n) - 1
        non_neighbors_mask = all_mask & ~neighbors_mask
        if has_i4_fast(adj, non_neighbors_mask):
            return False
        return True

    def is_valid(adj, n):
        """Check if the graph has no K5 and no I5."""
        all_mask = (1 << n) - 1
        # Check for K5: for each vertex, its neighborhood must have no K4
        for u in range(n):
            if has_k4_fast(adj, adj[u]):
                return False
        # Check for I5: for each vertex, its non-neighborhood must have no I4
        for u in range(n):
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_fast(adj, non_neigh):
                return False
        return True

    # Verify seed
    if not is_valid(adj, n):
        return n, []

    best_n = n
    best_adj = adj[:]

    # 2. Try to extend using smart heuristics
    current_n = n
    current_adj = adj[:]
    
    # Strategy: try adding vertices with carefully chosen neighbor sets
    # For balanced Ramsey graphs, degree should be around n/2
    
    max_target = 43  # We want to reach 42 if possible
    
    for target_n in range(current_n + 1, max_target + 1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Generate candidate neighbor set sizes
        mid = current_n // 2
        size_candidates = []
        for offset in range(0, current_n + 1):
            s1 = mid - offset
            s2 = mid + offset
            if 0 <= s1 <= current_n:
                size_candidates.append(s1)
            if 0 <= s2 <= current_n and s2 != s1:
                size_candidates.append(s2)
        size_candidates = list(set(size_candidates))
        random.shuffle(size_candidates)
        
        # Limit attempts per size to keep time reasonable
        max_attempts = 200
        
        for size in size_candidates:
            if time.time() > deadline:
                break
            if found:
                break
            if size < 0 or size > current_n:
                continue
                
            attempts = max_attempts
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
            # Try a few restarts with different random seeds
            restarts = 3
            for r in range(restarts):
                if time.time() > deadline:
                    break
                # Reset to best state
                current_n = best_n
                current_adj = best_adj[:]
                
                # Try with different random choices
                for size in size_candidates[:15]:
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    if size < 0 or size > current_n:
                        continue
                        
                    attempts = 300
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
                # One more aggressive restart with more attempts on balanced sizes
                current_n = best_n
                current_adj = best_adj[:]
                mid = current_n // 2
                balanced_sizes = [s for s in range(max(0, mid-3), min(current_n+1, mid+4))]
                random.shuffle(balanced_sizes)
                
                for size in balanced_sizes:
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
