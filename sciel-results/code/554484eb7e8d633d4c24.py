import numpy as np
import random
from itertools import combinations

def construct():
    n = 20
    
    # Use bitmasks for fast intersection and counting
    # masks[i] is an integer where bit j is 1 if edge (i, j) exists
    
    def bit_count(x):
        return x.bit_count()
        
    def get_set_bits(x):
        bits = []
        while x:
            lsb = x & -x
            bits.append(lsb.bit_length() - 1)
            x ^= lsb
        return bits

    def find_violations(masks):
        violations = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    common = masks[a] & masks[b] & masks[c]
                    if bit_count(common) >= 3:
                        violations.append((a, b, c, common))
        return violations

    def count_edges(masks):
        total = 0
        for i in range(n):
            total += bit_count(masks[i])
        return total // 2

    best_edges = []
    best_count = 0
    
    # Try multiple deterministic seeds
    for seed in range(20):
        random.seed(seed)
        
        # Initialize with a dense random graph
        # p=0.85 is a good starting point for K3,3 free search
        p = 0.85
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        
        # Local Search: Remove edges to fix violations
        # We iterate until no violations
        max_iter = 1000
        for _ in range(max_iter):
            violations = find_violations(masks)
            if not violations:
                break
            
            # Calculate blame for each edge
            # Edge (u, v) is blamed for each violation where u and v are connected 
            # and one of them is a common neighbor of the other two? 
            # Actually, if (a,b,c) is a violation with common neighbors C, 
            # removing (x, a) for x in C reduces the common neighbor count for (a,b,c) by 1.
            # So edge (x, a) is "involved".
            
            edge_blame = {}
            for a, b, c, common in violations:
                # Get list of common neighbors
                common_neighbors = get_set_bits(common)
                for x in common_neighbors:
                    # Blame (x, a), (x, b), (x, c)
                    # Note: x is a common neighbor, so x != a, b, c
                    # and edges (x,a), (x,b), (x,c) exist.
                    
                    u, v = (x, a) if x < a else (a, x)
                    key = (u, v)
                    edge_blame[key] = edge_blame.get(key, 0) + 1
                    
                    u, v = (x, b) if x < b else (b, x)
                    key = (u, v)
                    edge_blame[key] = edge_blame.get(key, 0) + 1
                    
                    u, v = (x, c) if x < c else (c, x)
                    key = (u, v)
                    edge_blame[key] = edge_blame.get(key, 0) + 1
            
            if not edge_blame:
                break
            
            # Pick the edge with max blame
            worst_edge = max(edge_blame, key=edge_blame.get)
            u, v = worst_edge
            # Remove edge
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            
        # Greedy Add: Try to add back missing edges
        # Generate list of missing edges
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] & (1 << j)):
                    missing_edges.append((i, j))
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            # Tentatively add
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
            # Check validity
            # Only need to check triples involving u or v
            valid = True
            
            # Check triples (u, b, c)
            for b in range(n):
                if b == u: continue
                for c in range(b + 1, n):
                    if c == u: continue
                    if c == b: continue
                    # Ensure distinct
                    if b == c: continue
                    # Actually, the loop structure above might generate duplicates or invalid indices if not careful.
                    # Let's just check all triples involving u or v simply.
                    pass
            
            # Simpler: Check all triples involving u
            # Triple (u, b, c) where b, c in [0..n-1], b != u, c != u, b != c
            for b in range(n):
                if b == u: continue
                for c in range(b + 1, n):
                    if c == u: continue
                    # b and c are distinct and not u
                    common = masks[u] & masks[b] & masks[c]
                    if bit_count(common) >= 3:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check triples (v, b, c) where b, c != v, b != c, and not (u, v, w) which is covered?
                # Triple (v, b, c). If b=u, it's (v, u, c) which is (u, v, c), covered above.
                # So we need to check triples where v is in the triple, but u is NOT in the triple?
                # No, adding (u, v) affects N(u) and N(v).
                # Common neighbors of (a, b, c) change only if a, b, or c is u or v.
                # We checked all triples with u.
                # Now check triples with v that do NOT contain u.
                for b in range(n):
                    if b == v or b == u: continue
                    for c in range(b + 1, n):
                        if c == v or c == u: continue
                        common = masks[v] & masks[b] & masks[c]
                        if bit_count(common) >= 3:
                            valid = False
                            break
                    if not valid:
                        break
            
            if not valid:
                # Revert
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            
        current_count = count_edges(masks)
        if current_count > best_count:
            best_count = current_count
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if masks[i] & (1 << j):
                        best_edges.append([i, j])
                        
    return best_edges
