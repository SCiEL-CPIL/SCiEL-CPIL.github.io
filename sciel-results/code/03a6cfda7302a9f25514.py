"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

N = 16
DEADLINE = 82  # safety margin before 90s kill
MASK_ALL = (1 << N) - 1

def _has_diameter2_bitset(adj_bits):
    """Check if graph has diameter exactly 2 using bitset representation."""
    n = N
    has_dist2 = False
    for a in range(n):
        for b in range(a + 1, n):
            if adj_bits[a] & (1 << b):
                continue
            common = adj_bits[a] & adj_bits[b]
            if common == 0:
                return False
            has_dist2 = True
    return has_dist2

def _all_edges_essential_bitset(adj_bits):
    """Check if every edge is essential using bitset representation."""
    n = N
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bits[u] & (1 << v)):
                continue
            mask_u = adj_bits[u] & ~(1 << v)
            mask_v = adj_bits[v] & ~(1 << u)
            
            for a in range(n):
                new_adj_a = adj_bits[a]
                if a == u:
                    new_adj_a = mask_u
                elif a == v:
                    new_adj_a = mask_v
                
                for b in range(a + 1, n):
                    if new_adj_a & (1 << b):
                        continue
                    new_adj_b = adj_bits[b]
                    if b == u:
                        new_adj_b = mask_u
                    elif b == v:
                        new_adj_b = mask_v
                    
                    common = new_adj_a & new_adj_b
                    if common == 0:
                        break
                else:
                    continue
                break
            else:
                return False
    return True

def _validate_full_bitset(adj_bits):
    """Full validation: diameter exactly 2 and all edges essential."""
    if not _has_diameter2_bitset(adj_bits):
        return False
    if not _all_edges_essential_bitset(adj_bits):
        return False
    return True

def _edges_from_bitset(adj_bits):
    """Convert bitset adjacency to edge list."""
    n = N
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_bits[u] & (1 << v):
                edges.append([u, v])
    return edges

def _make_k88():
    """Create K_{8,8} bitset: vertices 0-7 connected to 8-15."""
    adj = [0] * N
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def construct():
    start = time.time()
    best_edges = None
    best_count = -1
    
    # Start with K_{8,8}
    adj_k88 = _make_k88()
    if _validate_full_bitset(adj_k88):
        edges = _edges_from_bitset(adj_k88)
        best_count = len(edges)
        best_edges = edges
    
    def try_expand(adj_bits, time_budget):
        """Try to add edges while maintaining all-essential property using bitsets."""
        n = N
        adj_bits = list(adj_bits)
        
        if not _validate_full_bitset(adj_bits):
            return None
        
        pairs = list(combinations(range(n), 2))
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for (u, v) in pairs:
                if time.time() > time_budget:
                    break
                if adj_bits[u] & (1 << v):
                    continue
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                if _validate_full_bitset(adj_bits):
                    improved = True
                else:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
        return adj_bits
    
    # Phase 1: Greedy expansion from K_{8,8}
    time_budget = start + 30
    adj_exp = try_expand(adj_k88, time_budget)
    if adj_exp is not None:
        edges = _edges_from_bitset(adj_exp)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Phase 2: Try multiple random orderings for expansion
    import random
    remaining_time = start + DEADLINE - time.time()
    if remaining_time > 5:
        # Try different random permutations of edge additions
        for trial in range(3):
            if time.time() > start + DEADLINE - 3:
                break
            
            adj_trial = list(adj_k88)
            pairs = list(combinations(range(N), 2))
            random.shuffle(pairs)
            
            improved = True
            while improved and time.time() < start + DEADLINE - 3:
                improved = False
                for (u, v) in pairs:
                    if time.time() > start + DEADLINE - 3:
                        break
                    if adj_trial[u] & (1 << v):
                        continue
                    adj_trial[u] |= (1 << v)
                    adj_trial[v] |= (1 << u)
                    if _validate_full_bitset(adj_trial):
                        improved = True
                    else:
                        adj_trial[u] &= ~(1 << v)
                        adj_trial[v] &= ~(1 << u)
            
            edges = _edges_from_bitset(adj_trial)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    # Phase 3: Try starting from different base structures
    if time.time() < start + DEADLINE - 5:
        # Try K_{7,9} or other bipartite splits
        for split in [(6, 10), (7, 9), (5, 11)]:
            if time.time() > start + DEADLINE - 3:
                break
            a, b = split
            adj_split = [0] * N
            for u in range(a):
                for v in range(a, a + b):
                    adj_split[u] |= (1 << v)
                    adj_split[v] |= (1 << u)
            
            if _validate_full_bitset(adj_split):
                adj_exp2 = try_expand(adj_split, start + DEADLINE - 3)
                if adj_exp2 is not None:
                    edges = _edges_from_bitset(adj_exp2)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = edges
    
    # Fallback
    if best_edges is None:
        best_edges = _edges_from_bitset(adj_k88)
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
