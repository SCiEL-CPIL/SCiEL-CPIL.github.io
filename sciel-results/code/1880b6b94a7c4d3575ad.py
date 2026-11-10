import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 18
    # Initial graph: K6,6,6
    edges = set()
    parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.add((min(u, v), max(u, v)))
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_triangles():
        tris = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            mask = (1 << u) | (1 << v) | (1 << w)
                            tris.append(mask)
        return tris

    def get_score(tris):
        # Check masks in decreasing order of popcount
        # Precompute masks by size?
        # Or just generate combinations?
        # Using combinations is slow.
        # Bitmask iteration is faster.
        
        # We can precompute a list of all masks sorted by popcount descending?
        # 2^18 = 262144 masks. Sorting once is fine.
        # But we need to do this inside the loop? No, precompute outside.
        
        pass

    # Precompute all masks sorted by popcount descending
    all_masks = list(range(1 << n))
    all_masks.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    def compute_score(tris):
        for m in all_masks:
            is_tf = True
            for t in tris:
                if (m & t) == t:
                    is_tf = False
                    break
            if is_tf:
                return bin(m).count('1')
        return 0

    current_score = compute_score(get_triangles())
    best_score = current_score
    best_edges = set(edges)
    current_tris = get_triangles()
    
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        add_edge = (u, v) not in edges
        
        if add_edge:
            common = adj[u] & adj[v]
            conflict = False
            for x in common:
                for y in common:
                    if x == y:
                        continue
                    if (min(x, y), max(x, y)) in edges:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
        
        # Apply
        if add_edge:
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            # Update triangles
            # New triangles involving u, v
            for w in common:
                mask = (1 << u) | (1 << v) | (1 << w)
                current_tris.append(mask)
        else:
            edges.remove((u, v))
            adj[u].remove(v)
            adj[v].remove(u)
            # Remove triangles involving u, v
            # This is hard to do incrementally correctly if we just appended.
            # We should rebuild triangles if removing.
            # Or, we can just rebuild triangles every time?
            # Rebuilding is O(E * avg_degree) ~ O(n^3) = 5832 ops. Fast.
            # So let's just rebuild triangles for simplicity and correctness.
            current_tris = get_triangles()

        new_score = compute_score(current_tris)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(edges)
            
        if new_score < current_score:
            current_score = new_score
        else:
            if random.random() > pow(2.718, -(new_score - current_score) / T):
                # Revert
                if add_edge:
                    edges.remove((u, v))
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                current_tris = get_triangles()
            else:
                current_score = new_score
                
        T *= cooling

    return [list(e) for e in best_edges]
