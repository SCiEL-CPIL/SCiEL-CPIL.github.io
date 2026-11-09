import random
import itertools

def is_valid(edges, n):
    """Check if no simple 3-edge path has endpoints of equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]
    
    # Group vertices by degree
    by_degree = {}
    for i in range(n):
        by_degree.setdefault(degrees[i], []).append(i)
    
    # Only check pairs with same degree
    for deg, verts in by_degree.items():
        for i, s in enumerate(verts):
            for t in verts[i+1:]:
                # Check if there's a simple path s-u-v-t of length 3
                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t or v == u:
                            continue
                        if t in adj[v]:
                            return False
    return True

def havel_hakimi(degrees):
    """Check if degree sequence is graphical using Havel-Hakimi."""
    d = sorted(degrees, reverse=True)
    if any(x < 0 or x > len(d) - 1 for x in d):
        return False
    while d and d[0] > 0:
        k = d[0]
        if k >= len(d):
            return False
        d = d[1:]
        for i in range(k):
            d[i] -= 1
            if d[i] < 0:
                return False
        d.sort(reverse=True)
    return True

def construct_graph_from_degrees(n, target_degrees):
    """Try to construct a graph with given degree sequence."""
    # Use greedy algorithm
    degrees = list(target_degrees)
    adj = [set() for _ in range(n)]
    
    # Sort vertices by target degree descending
    order = sorted(range(n), key=lambda i: -target_degrees[i])
    
    for i in order:
        remaining = target_degrees[i]
        if remaining <= 0:
            continue
        # Connect to vertices with highest remaining degree (excluding self)
        candidates = [j for j in range(n) if j != i and j not in adj[i]]
        candidates.sort(key=lambda j: -target_degrees[j] + len(adj[j]))
        
        for j in candidates:
            if remaining <= 0:
                break
            if j in adj[i]:
                continue
            adj[i].add(j)
            adj[j].add(i)
            remaining -= 1
    
    # Check if we achieved the target degrees
    for i in range(n):
        if len(adj[i]) != target_degrees[i]:
            return None
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges

def local_search_add(edges, n, max_iterations=5000):
    """Try to add edges to increase count while maintaining validity."""
    current_edges = list(edges)
    current_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]
    
    for _ in range(max_iterations):
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) in current_set:
                    continue
                # Try adding edge (u,v)
                adj[u].add(v)
                adj[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                
                if is_valid(current_edges + [[u,v]], n):
                    current_edges.append([u,v])
                    current_set.add((u,v))
                    improved = True
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    degrees[u] -= 1
                    degrees[v] -= 1
        
        if not improved:
            break
    
    return current_edges

def construct():
    n = 15
    best_edges = []
    best_count = 0
    
    # Strategy 1: Try to construct graphs with distinct degrees
    # Degrees must be distinct, sum must be even, and graphical
    
    # Try various degree sequences with all distinct values
    # Max possible sum with 15 distinct degrees from 0-14: 105 (odd, so adjust)
    
    # Try degree sequences close to 0,1,2,...,14 but with even sum
    # Remove one pair and adjust: e.g., use degrees that sum to 104 or 102
    
    candidate_degree_seqs = []
    
    # Base: 0,1,2,...,14 sums to 105. Need even sum.
    # Try removing degree 0 and 14, replacing with adjusted values
    # Or just try many random distinct degree sequences
    
    import random
    random.seed(42)
    
    # Generate candidate degree sequences with all distinct degrees
    for trial in range(200):
        # Pick 15 distinct degrees from 0 to 14
        base = list(range(15))
        # Randomly perturb to get different sequences
        if trial < 50:
            # Use permutations of 0-14
            seq = base[:]
            random.shuffle(seq)
        elif trial < 150:
            # Try sequences with one pair adjusted
            seq = base[:]
            i, j = random.sample(range(15), 2)
            # Adjust to keep distinct and sum even
            if (sum(seq) % 2) == 1:
                # Need to make sum even
                # Decrease one odd-degree vertex by 1 or increase one even by 1
                for k in range(15):
                    if seq[k] > 0 and seq[k] - 1 not in seq:
                        seq[k] -= 1
                        break
                else:
                    for k in range(15):
                        if seq[k] < 14 and seq[k] + 1 not in seq:
                            seq[k] += 1
                            break
            seq = sorted(seq, reverse=True)
        else:
            # Random distinct degrees
            seq = random.sample(range(15), 15)
            if sum(seq) % 2 == 1:
                # Adjust
                for k in range(15):
                    if seq[k] > 0 and (seq[k]-1) not in seq:
                        seq[k] -= 1
                        break
                else:
                    seq = random.sample(range(14, 0, -1), 15) if False else seq
                    continue
            seq = sorted(seq, reverse=True)
        
        if len(set(seq)) != 15:
            continue
        if sum(seq) % 2 != 0:
            continue
        
        if havel_hakimi(seq):
            candidate_degree_seqs.append(seq)
    
    # Sort by sum descending (more edges)
    candidate_degree_seqs.sort(key=lambda x: -sum(x))
    
    for seq in candidate_degree_seqs[:50]:
        # Try to construct graph with this degree sequence
        for attempt in range(10):
            edges = construct_graph_from_degrees(n, seq)
            if edges is not None and is_valid(edges, n):
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = list(edges)
                break
    
    # Strategy 2: Local search to add edges
    if best_edges:
        improved_edges = local_search_add(best_edges, n, max_iterations=10000)
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
    
    # Strategy 3: Also try K_7,8 as baseline
    k78_edges = [[u, v] for u in range(7) for v in range(7, 15)]
    if is_valid(k78_edges, n):
        if len(k78_edges) > best_count:
            best_count = len(k78_edges)
            best_edges = list(k78_edges)
            improved_edges = local_search_add(best_edges, n, max_iterations=10000)
            if len(improved_edges) > best_count:
                best_count = len(improved_edges)
                best_edges = improved_edges
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        u, v = e
        if not isinstance(u, int) or isinstance(u, bool):
            u = int(u)
        if not isinstance(v, int) or isinstance(v, bool):
            v = int(v)
        result.append([u, v])
    
    return result
