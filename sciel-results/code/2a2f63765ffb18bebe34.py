"""Rewritten program using constraint propagation with conflict maximization."""
import itertools
import random
import time

def _is_regular_subgraph(vertices, adj_bits):
    """Check if the induced subgraph on 'vertices' is regular using bitmasks."""
    k = len(vertices)
    if k < 2:
        return True
    
    mask = 0
    for v in vertices:
        mask |= (1 << v)
        
    degs = []
    for v in vertices:
        d = bin(adj_bits[v] & mask).count('1')
        degs.append(d)
        
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def construct():
    def build_and_optimize(n, max_time=110):
        start_time = time.time()
        
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        
        subsets6 = list(itertools.combinations(range(n), 6))
        num_subsets = len(subsets6)
        
        # Precompute masks for subsets
        subset_masks = []
        for s in subsets6:
            mask = 0
            for v in s:
                mask |= (1 << v)
            subset_masks.append(mask)
            
        # Precompute edge to subset indices map
        edge_map = {}
        for i, s in enumerate(subsets6):
            for a in range(len(s)):
                for b in range(a+1, len(s)):
                    u, v = s[a], s[b]
                    key = (min(u,v), max(u,v))
                    if key not in edge_map:
                        edge_map[key] = []
                    edge_map[key].append(i)
        
        # Initialize violations
        violations = set()
        for i in range(num_subsets):
            if _is_regular_subgraph(subsets6[i], adj_bits):
                violations.add(i)
        
        for iteration in range(100000):
            if time.time() - start_time > max_time:
                break
            
            if not violations:
                if _verify_no_large_regular(adj_bits, n):
                    return adj_bits, True
                else:
                    large_viol = _find_large_regular(adj_bits, n)
                    if large_viol:
                        edges_in_large = [(large_viol[i], large_viol[j]) for i in range(len(large_viol)) for j in range(i+1, len(large_viol))]
                        u, v = random.choice(edges_in_large)
                        adj_bits[u] ^= (1 << v)
                        adj_bits[v] ^= (1 << u)
                        
                        key = (min(u,v), max(u,v))
                        if key in edge_map:
                            for idx in edge_map[key]:
                                if _is_regular_subgraph(subsets6[idx], adj_bits):
                                    violations.add(idx)
                                else:
                                    violations.discard(idx)
                    continue
            
            edge_viol_count = {}
            for idx in violations:
                s = subsets6[idx]
                for a in range(len(s)):
                    for b in range(a+1, len(s)):
                        u, v = s[a], s[b]
                        key = (min(u,v), max(u,v))
                        edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
                        
            if not edge_viol_count:
                break
                
            sorted_edges = sorted(edge_viol_count.items(), key=lambda x: -x[1])
            
            best_flip = None
            best_score = -1
            
            for (u, v), count in sorted_edges[:20]:
                key = (min(u,v), max(u,v))
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                fixed = 0
                if key in edge_map:
                    for idx in edge_map[key]:
                        if idx in violations:
                            if not _is_regular_subgraph(subsets6[idx], adj_bits):
                                fixed += 1
                        else:
                            if _is_regular_subgraph(subsets6[idx], adj_bits):
                                fixed -= 1
                
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                if fixed > best_score:
                    best_score = fixed
                    best_flip = (u, v)
                    
            if best_flip and best_score > 0:
                u, v = best_flip
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                key = (min(u,v), max(u,v))
                if key in edge_map:
                    for idx in edge_map[key]:
                        if _is_regular_subgraph(subsets6[idx], adj_bits):
                            violations.add(idx)
                        else:
                            violations.discard(idx)
            else:
                idx = random.choice(list(violations))
                s = subsets6[idx]
                u, v = random.choice([(s[a], s[b]) for a in range(len(s)) for b in range(a+1, len(s))])
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                key = (min(u,v), max(u,v))
                if key in edge_map:
                    for idx2 in edge_map[key]:
                        if _is_regular_subgraph(subsets6[idx2], adj_bits):
                            violations.add(idx2)
                        else:
                            violations.discard(idx2)
                            
        return adj_bits, False
    
    def _verify_no_large_regular(adj_bits, n):
        for k in range(7, n+1):
            for subset in itertools.combinations(range(n), k):
                if _is_regular_subgraph(subset, adj_bits):
                    return False
        return True
    
    def _find_large_regular(adj_bits, n):
        for k in range(7, n+1):
            for subset in itertools.combinations(range(n), k):
                if _is_regular_subgraph(subset, adj_bits):
                    return subset
        return None
    
    for n in [21, 20]:
        adj_bits, success = build_and_optimize(n, max_time=55)
        if success:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_bits[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return {'n': 5, 'edges': []}
