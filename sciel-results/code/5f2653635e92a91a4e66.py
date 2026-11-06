import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n = 16
    adj = [0] * n
    
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_edges = n * (n - 1) // 2
    best_edges = 0
    best_adj = adj[:]

    # Precompute Q3 patterns
    q3_edges_canonical = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                q3_edges_canonical.append((u, v))
                
    q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        edges = frozenset(tuple(sorted((perm[u], perm[v]))) for u, v in q3_edges_canonical)
        q3_patterns.add(edges)
        
    def _check_subset_q3(adj_list, subset, patterns):
        for pattern in patterns:
            match = True
            for u, v in pattern:
                if not ((adj_list[subset[u]] >> subset[v]) & 1):
                    match = False
                    break
            if match:
                return True
        return False

    def has_q3(adj_list):
        # Shuffle subsets to randomize search
        subsets = list(itertools.combinations(range(n), 8))
        random.shuffle(subsets)
        
        for subset in subsets:
            # Quick check: edge count
            # If < 12, skip
            ec = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if (adj_list[subset[i]] >> subset[j]) & 1:
                        ec += 1
            if ec < 12:
                continue
                
            if _check_subset_q3(adj_list, subset, q3_patterns):
                # Return edges
                # We need to find which pattern matched to return edges?
                # For removal, we just need ONE edge from the Q3.
                # We can just return the first edge of the first matching pattern.
                # But _check_subset_q3 doesn't return the pattern.
                # Let's modify _check_subset_q3 to return the pattern or None.
                pass
        return None

    # Modified _check_subset_q3 to return pattern
    def _find_q3_pattern(adj_list, subset, patterns):
        for pattern in patterns:
            match = True
            for u, v in pattern:
                if not ((adj_list[subset[u]] >> subset[v]) & 1):
                    match = False
                    break
            if match:
                return pattern
        return None

    def has_q3_with_edges(adj_list):
        subsets = list(itertools.combinations(range(n), 8))
        random.shuffle(subsets)
        
        for subset in subsets:
            ec = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if (adj_list[subset[i]] >> subset[j]) & 1:
                        ec += 1
            if ec < 12:
                continue
                
            pattern = _find_q3_pattern(adj_list, subset, q3_patterns)
            if pattern:
                # Convert pattern to edges in original graph
                edges = [[subset[u], subset[v]] for u, v in pattern]
                return edges
        return None

    def try_add_edges(adj_list):
        added = 0
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_list[u] >> v) & 1):
                    non_edges.append((u, v))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            adj_list[u] |= (1 << v)
            adj_list[v] |= (1 << u)
            
            # Check if new edge creates Q3
            # Only check subsets containing u and v
            found = False
            other_vertices = [x for x in range(n) if x not in (u, v)]
            for subset6 in itertools.combinations(other_vertices, 6):
                full_subset = tuple(sorted([u, v] + list(subset6)))
                if _find_q3_pattern(adj_list, full_subset, q3_patterns):
                    found = True
                    break
            
            if found:
                adj_list[u] ^= (1 << v)
                adj_list[v] ^= (1 << u)
            else:
                added += 1
                
        return added

    while time.time() < deadline - 10:
        q3_edges = has_q3_with_edges(adj)
        
        if q3_edges:
            u, v = q3_edges[0]
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edges -= 1
        else:
            # Try to add edges
            added = try_add_edges(adj)
            current_edges += added
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                
            if added == 0:
                # Perturb
                if current_edges > 0:
                    while True:
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u != v and ((adj[u] >> v) & 1):
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            current_edges -= 1
                            break

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
