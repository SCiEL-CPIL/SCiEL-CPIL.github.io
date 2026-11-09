"""Packing trees T1..T20 into K20 using combined random-permutation + BFS approach."""
import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def construct():
    """
    Embed trees T1..T20 into K20 with edge-disjoint embeddings.
    Returns a list of 20 entries; entry k-1 is None or a list of k distinct ints in 0..19.
    Position i is the image of vertex i of T_k.
    """
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n = 20
    max_possible = sum(len(t) for t in trees)
    
    # Pre-compute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges_k = trees[k-1]
        adj = [[] for _ in range(k)]
        for (u, v) in edges_k:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    def try_random_permutation(rng, k, edges_k, used_edges):
        """Try random permutations to find valid embedding."""
        # Number of attempts scales with tree size
        num_tries = max(100, 20000 // k)
        for _ in range(num_tries):
            mapping = rng.sample(range(n), k)
            valid = True
            for (u, v) in edges_k:
                key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                if key in used_edges:
                    valid = False
                    break
            if valid:
                return mapping
        return None
    
    def try_bfs_embedding(rng, k, edges_k, adj_k, used_edges):
        """BFS-guided embedding with smart candidate selection."""
        for attempt in range(200):
            mapping = [None] * k
            root = rng.randint(0, n - 1)
            mapping[0] = root
            visited = [False] * k
            visited[0] = True
            queue = deque([0])
            used_hosts = {root}
            valid_tree = True
            
            while queue:
                node = queue.popleft()
                for neighbor in adj_k[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent_host = mapping[node]
                        candidates = []
                        for h in range(n):
                            if h in used_hosts:
                                continue
                            key = (min(parent_host, h), max(parent_host, h))
                            if key not in used_edges:
                                candidates.append(h)
                        if candidates:
                            # Score candidates by how many free edges they have (prefer less constrained)
                            def score(h):
                                free = 0
                                for nh in range(n):
                                    if nh in used_hosts or nh == h:
                                        continue
                                    nk = (min(h, nh), max(h, nh))
                                    if nk not in used_edges:
                                        free += 1
                                return free
                            scored = [(score(h), h) for h in candidates]
                            rng.shuffle(scored)
                            # Pick randomly among top-scoring for diversity
                            top_score = max(s for s, _ in scored)
                            top_candidates = [h for s, h in scored if s >= top_score - 2]
                            mapping[neighbor] = rng.choice(top_candidates)
                            used_hosts.add(mapping[neighbor])
                            queue.append(neighbor)
                        else:
                            valid_tree = False
                            break
                if not valid_tree:
                    break
            
            if valid_tree and all(mapping[i] is not None for i in range(k)):
                # Final verification
                ok = True
                for (u, v) in edges_k:
                    key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    if key in used_edges:
                        ok = False
                        break
                if ok:
                    return mapping
        return None
    
    def try_backtrack_embedding(rng, k, edges_k, adj_k, used_edges, max_nodes=15):
        """Backtracking embedding for small-to-medium trees."""
        if k > max_nodes:
            return None
        for attempt in range(50):
            mapping = [None] * k
            # Order vertices by degree (high degree first)
            deg = [len(adj_k[i]) for i in range(k)]
            order = sorted(range(k), key=lambda i: -deg[i])
            # Build parent-child relationships based on order
            # We'll use a simple recursive backtracking
            def backtrack(idx, used_hosts_set):
                if idx == k:
                    return True
                v = order[idx]
                # Find valid hosts
                valid_hosts = []
                for h in range(n):
                    if h in used_hosts_set:
                        continue
                    # Check all edges from v to already-mapped neighbors
                    ok = True
                    for nb in adj_k[v]:
                        if mapping[nb] is not None:
                            key = (min(h, mapping[nb]), max(h, mapping[nb]))
                            if key in used_edges:
                                ok = False
                                break
                    if ok:
                        valid_hosts.append(h)
                rng.shuffle(valid_hosts)
                for h in valid_hosts:
                    mapping[v] = h
                    used_hosts_set.add(h)
                    if backtrack(idx + 1, used_hosts_set):
                        return True
                    mapping[v] = None
                    used_hosts_set.discard(h)
                return False
            
            if backtrack(0, set()):
                # Final verification
                ok = True
                for (u, v) in edges_k:
                    key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    if key in used_edges:
                        ok = False
                        break
                if ok:
                    return list(mapping)
        return None
    
    def solve_one_restart(rng, order):
        used_edges = set()
        result = [None] * n
        
        for k in order:
            edges_k = trees[k-1]
            if not edges_k:
                result[k-1] = [0]
                continue
            if k == 2:
                found = False
                for _ in range(10000):
                    a = rng.randint(0, n - 1)
                    b = rng.randint(0, n - 1)
                    if a != b:
                        key = (min(a, b), max(a, b))
                        if key not in used_edges:
                            result[k-1] = [a, b]
                            used_edges.add(key)
                            found = True
                            break
                if not found:
                    for a in range(n):
                        for b in range(a + 1, n):
                            key = (a, b)
                            if key not in used_edges:
                                result[k-1] = [a, b]
                                used_edges.add(key)
                                found = True
                                break
                        if found:
                            break
                if not found:
                    return None
                continue
            
            adj_k = tree_adj[k-1]
            
            # Strategy 1: Random permutation (fast for small trees)
            best = try_random_permutation(rng, k, edges_k, used_edges)
            
            # Strategy 2: BFS guided (better for larger trees)
            if best is None:
                best = try_bfs_embedding(rng, k, edges_k, adj_k, used_edges)
            
            # Strategy 3: Backtracking (for small trees where BFS fails)
            if best is None and k <= 15:
                best = try_backtrack_embedding(rng, k, edges_k, adj_k, used_edges)
            
            if best is None:
                return None
            
            result[k-1] = list(best)
            for (u, v) in edges_k:
                used_edges.add((min(best[u], best[v]), max(best[u], best[v])))
        
        return result
    
    # Generate orderings
    orderings = [
        list(range(n, 0, -1)),  # largest first
        list(range(1, n + 1)),  # smallest first
    ]
    # Alternating
    alt = []
    left, right = 0, n - 1
    while left <= right:
        if left == right:
            alt.append(left + 1)
            break
        alt.append(left + 1)
        alt.append(right + 1)
        left += 1
        right -= 1
    orderings.append(alt)
    # Every other: small, large, small, large starting from middle
    mid_order = []
    mid = n // 2
    for i in range(mid):
        mid_order.append(i + 1)
        mid_order.append(n - i)
    if mid * 2 < n:
        mid_order.append(mid + 1)
    orderings.append(mid_order)
    # Degree-based: embed trees with highest max-degree first
    max_degrees = []
    for k in range(1, n + 1):
        adj_k = tree_adj[k-1]
        max_d = max(len(adj_k[i]) for i in range(k)) if k > 1 else 0
        max_degrees.append((max_d, k))
    max_degrees.sort(key=lambda x: (-x[0], -x[1]))
    orderings.append([k for _, k in max_degrees])
    
    best_solution = None
    best_value = -1
    
    restart = 0
    while time.time() < deadline:
        order_idx = restart % len(orderings)
        order = list(orderings[order_idx])
        rng = Random(restart * 7919 + 13)
        
        # Perturbation for diversity
        if rng.random() < 0.4 and len(order) > 2:
            i = rng.randint(1, len(order) - 2)
            order[i], order[i+1] = order[i+1], order[i]
        elif rng.random() < 0.15 and len(order) > 3:
            # Swap non-adjacent elements
            i = rng.randint(0, len(order) - 2)
            j = rng.randint(i + 2, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        solution = solve_one_restart(rng, order)
        if solution is not None:
            val = sum(len(trees[k-1]) for k in range(1, n + 1) if solution[k-1] is not None)
            if val > best_value:
                best_value = val
                best_solution = [list(s) if s is not None else None for s in solution]
        
        if best_value >= max_possible:
            break
        restart += 1
    
    if best_solution is None:
        return [None] * 19 + [list(range(n))]
    
    return best_solution


if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k-1] is not None:
            r = res[k-1]
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v}")
                    ok = False
            for (u, v) in trees[k-1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k-1]) for k in range(1, 21) if res[k-1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
