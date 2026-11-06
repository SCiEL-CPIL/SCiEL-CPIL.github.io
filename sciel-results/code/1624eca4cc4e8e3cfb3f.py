"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    n = 18
    
    # Use bitmasks for fast operations
    # adj_mask[v] is a bitmask of neighbors of vertex v
    # Start with complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj_mask = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
    
    def is_k4_free(masks):
        # Check no K4 using bitmasks
        # A K4 exists iff there are 4 vertices all pairwise connected
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] >> b) & 1:
                    continue
                # Common neighbors of a and b
                common = masks[a] & masks[b]
                # Remove a and b from common
                common &= ~(1 << a)
                common &= ~(1 << b)
                # Check if any two vertices in common are connected
                c_bits = common
                while c_bits:
                    c = c_bits & (-c_bits)
                    c_idx = c.bit_length() - 1
                    c_bits ^= c
                    # Check if any vertex in common after c is connected to c
                    d_bits = common & ~(c - 1) & ~c
                    # Actually we need to check if c and d are both in common and connected
                    # Simpler: check if (masks[c_idx] & common & ~(c-1) & ~c) != 0
                    if (masks[c_idx] & common & ~(c - 1) & ~c) != 0:
                        return False
        return True
    
    def max_triangle_free_induced(masks):
        # Find largest subset of vertices inducing a triangle-free graph
        # A set S is triangle-free iff no three vertices in S are all mutually adjacent
        # Use branch and bound with bitmasks
        
        best = 0
        all_vertices = (1 << n) - 1
        
        # Precompute for each vertex, its neighbors as bitmask
        # neighbors[i] = masks[i]
        
        # Branch and bound: build up a triangle-free set
        # State: (current_mask, candidates_mask)
        # candidates_mask contains vertices that can be added without creating a triangle
        
        # A vertex v can be added to current_set if v's neighbors in current_set form an independent set
        # i.e., no two neighbors of v in current_set are connected to each other
        
        def can_add(current_mask, v):
            # v's neighbors in current_mask
            v_neigh_in_current = masks[v] & current_mask
            # Check if v_neigh_in_current is independent
            # For each u in v_neigh_in_current, check if u's neighbors intersect v_neigh_in_current
            bits = v_neigh_in_current
            while bits:
                b = bits & (-bits)
                u = b.bit_length() - 1
                bits ^= b
                if (masks[u] & v_neigh_in_current & ~b) != 0:
                    return False
            return True
        
        def search(current_mask, candidates_mask, depth):
            nonlocal best
            if time.time() > deadline - 5:
                return
            # Upper bound: current size + popcount(candidates)
            if current_mask.bit_count() + candidates_mask.bit_count() <= best:
                return
            if candidates_mask == 0:
                if current_mask.bit_count() > best:
                    best = current_mask.bit_count()
                return
            
            # Pick a candidate - use the one with highest degree in candidates for better pruning
            # Or just pick the lowest bit
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask ^ v_bit
            
            # Option 1: include v if possible
            if can_add(current_mask, v):
                new_current = current_mask | v_bit
                # Filter remaining: only those that can be added to new_current
                new_cands = 0
                temp = remaining
                while temp:
                    b = temp & (-temp)
                    idx = b.bit_length() - 1
                    temp ^= b
                    if can_add(new_current, idx):
                        new_cands |= b
                search(new_current, new_cands, depth + 1)
            
            # Option 2: exclude v
            search(current_mask, remaining, depth + 1)
        
        search(0, all_vertices, 0)
        return best
    
    # Verify initial graph
    if not is_k4_free(adj_mask):
        # Fallback - return complete tripartite
        result = []
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        result.append([a, b])
        return result
    
    current_best = max_triangle_free_induced(adj_mask)
    
    # Local search: try edge additions and removals
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
    
    random.seed(42)
    
    # Precompute all missing edges
    current_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if (adj_mask[u] >> v) & 1:
                current_edges.add((u, v))
    
    while time.time() < deadline - 10:
        # Try random edge modifications
        missing = all_edges - current_edges
        if missing:
            candidate_edge = random.choice(list(missing))
            u, v = candidate_edge
            # Check K4-free after adding edge (u,v)
            new_mask = adj_mask[:]
            new_mask[u] |= (1 << v)
            new_mask[v] |= (1 << u)
            if is_k4_free(new_mask):
                score = max_triangle_free_induced(new_mask)
                if score < current_best:
                    adj_mask = new_mask
                    current_edges.add(candidate_edge)
                    current_best = score
                    continue
        
        # Remove a random existing edge
        if current_edges:
            candidate_edge = random.choice(list(current_edges))
            u, v = candidate_edge
            new_mask = adj_mask[:]
            new_mask[u] &= ~(1 << v)
            new_mask[v] &= ~(1 << u)
            # Removing edges always preserves K4-free
            score = max_triangle_free_induced(new_mask)
            if score < current_best:
                adj_mask = new_mask
                current_edges.discard(candidate_edge)
                current_best = score
    
    # Convert adj_mask back to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj_mask[u] >> v) & 1:
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
