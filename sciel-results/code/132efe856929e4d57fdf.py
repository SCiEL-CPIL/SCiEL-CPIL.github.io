import time
import random
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 100.0

def _has_c5(n, adj_bits):
    """Check if graph has a C5 using bitmask approach."""
    for u in range(n):
        nu = adj_bits[u]
        for a in range(n):
            if a == u: continue
            if not (nu >> a) & 1: continue
            for c in range(n):
                if c == u or c == a: continue
                if not (adj_bits[a] >> c) & 1: continue
                common = adj_bits[a] & adj_bits[c]
                common &= ~(1 << u)
                common &= ~(1 << a)
                common &= ~(1 << c)
                if common:
                    return True
    return False

def _has_independent_set(n, adj_bits, target=9):
    """Check if graph has an independent set of size 'target' using backtracking."""
    def search(remaining, current_size):
        if current_size == target:
            return True
        if remaining.bit_count() < target - current_size:
            return False
        
        bits = []
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            bits.append(idx)
            temp ^= lsb
            
        if not bits:
            return False
            
        v = bits[0]
        
        new_remaining_incl = remaining & ~(1 << v) & ~adj_bits[v]
        if search(new_remaining_incl, current_size + 1):
            return True
            
        new_remaining_excl = remaining & ~(1 << v)
        if search(new_remaining_excl, current_size):
            return True
            
        return False

    return search((1 << n) - 1, 0)

def _add_edge_safe(n, adj_bits, u, v):
    """Try to add edge (u,v) without creating C5. Returns True if safe."""
    if u == v or (adj_bits[u] >> v) & 1:
        return True
    
    # Check if adding this edge creates a C5
    # A C5 containing edge (u,v) has form u-a-b-c-v-u
    # So we need: a adjacent to u, c adjacent to v, a adjacent to c, and common neighbor of a and c (excluding u,v,a,c)
    nu = adj_bits[u] & ~(1 << v)
    nv = adj_bits[v] & ~(1 << u)
    
    for a in range(n):
        if a == v: continue
        if not (nu >> a) & 1: continue
        for c in range(n):
            if c == u or c == a: continue
            if not (nv >> c) & 1: continue
            if not (adj_bits[a] >> c) & 1: continue
            common = adj_bits[a] & adj_bits[c]
            common &= ~(1 << u)
            common &= ~(1 << v)
            common &= ~(1 << a)
            common &= ~(1 << c)
            if common:
                return False
    return True

def _independence_number_approx(n, adj_bits):
    """Approximate independence number using greedy + local search."""
    # Greedy: repeatedly pick vertex with min degree in remaining graph
    remaining = (1 << n) - 1
    alpha = 0
    while remaining:
        # Find vertex with minimum degree in remaining subgraph
        min_deg = n + 1
        min_v = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj_bits[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp ^= lsb
        if min_v == -1:
            break
        alpha += 1
        remaining &= ~(1 << min_v)
        remaining &= ~adj_bits[min_v]
    return alpha

def construct():
    start_time = time.time()
    
    # Baseline: 8 disjoint K4s (n=32)
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
    
    # Try to find larger n using randomized greedy approach
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 30:
            break
        
        found = False
        for restart in range(100):
            if time.time() > _DEADLINE - 15:
                break
            if found:
                break
                
            n = target_n
            adj_bits = [0] * n
            
            # Strategy: Start with a sparse random graph, then add edges greedily
            # to reduce independence number while avoiding C5
            
            # Initialize with some random edges (sparse)
            random.seed(restart * 1000 + n)
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.1:
                        if _add_edge_safe(n, adj_bits, u, v):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
            
            # Now greedily add edges to reduce alpha
            # We want alpha < 9
            max_iterations = 500
            for it in range(max_iterations):
                if time.time() > _DEADLINE - 10:
                    break
                    
                # Check current alpha
                if _has_independent_set(n, adj_bits, 9):
                    # Need to reduce alpha - find a vertex to add edges from
                    # Pick a vertex that's in many independent sets
                    # Simple heuristic: pick a random vertex and try adding edges
                    v = random.randint(0, n-1)
                    
                    # Try adding edges from v to non-neighbors
                    candidates = []
                    for u in range(n):
                        if u != v and not (adj_bits[v] >> u) & 1:
                            candidates.append(u)
                    
                    random.shuffle(candidates)
                    added = False
                    for u in candidates:
                        if _add_edge_safe(n, adj_bits, v, u):
                            adj_bits[v] |= (1 << u)
                            adj_bits[u] |= (1 << v)
                            added = True
                            break
                    
                    if not added:
                        # Try removing an edge to allow future additions
                        # Find a random edge to remove
                        edges_to_consider = []
                        for u in range(n):
                            temp = adj_bits[u]
                            while temp:
                                lsb = temp & -temp
                                w = lsb.bit_length() - 1
                                if w > u:
                                    edges_to_consider.append((u, w))
                                temp ^= lsb
                        
                        if edges_to_consider:
                            u, w = random.choice(edges_to_consider)
                            adj_bits[u] &= ~(1 << w)
                            adj_bits[w] &= ~(1 << u)
                else:
                    # Alpha < 9, check for C5
                    if not _has_c5(n, adj_bits):
                        # Valid graph found!
                        found = True
                        best_n = n
                        best_edges = []
                        for u in range(n):
                            temp = adj_bits[u]
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                if v > u:
                                    best_edges.append([u, v])
                                temp ^= lsb
                        break
                    else:
                        # Has C5, need to remove some edges
                        # Find a C5 and break it
                        # For simplicity, just remove a random edge
                        edges_to_consider = []
                        for u in range(n):
                            temp = adj_bits[u]
                            while temp:
                                lsb = temp & -temp
                                v = lsb.bit_length() - 1
                                if v > u:
                                    edges_to_consider.append((u, v))
                                temp ^= lsb
                        
                        if edges_to_consider:
                            u, v = random.choice(edges_to_consider)
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                        else:
                            break
    
    # Final verification of best result
    if best_n >= 1:
        # Reconstruct adj_bits from best_edges for final check
        adj_bits = [0] * best_n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        if _has_c5(best_n, adj_bits) or _has_independent_set(best_n, adj_bits, 9):
            # Invalid, fall back to baseline
            best_n = 32
            best_edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        best_edges.append([base+i, base+j])
    
    return {'n': best_n, 'edges': best_edges}
