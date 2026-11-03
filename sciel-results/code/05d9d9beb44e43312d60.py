"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using BFS from each vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity and simplicity
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            if len(adj[u]) > k or len(adj[v]) > k:
                return False
        for i in range(n):
            if len(adj[i]) != k:
                return False

        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            q = deque([start])
            dist[start] = 0
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_regular(n, k, seed=None):
        if seed is not None:
            random.seed(seed)
        if n * k % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def count_short_cycles_heuristic(n, adj, limit):
        """
        Fast heuristic for counting short cycles.
        Counts triangles and 4-cycles exactly.
        """
        adj_set = [set(adj[i]) for i in range(n)]
        tri_count = 0
        four_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj_set[u] & adj_set[v])
                if common > 0:
                    tri_count += common
                if common >= 2:
                    four_count += common * (common - 1) // 2
        return tri_count + 4 * four_count

    def local_search_girth(n, edges, target_girth, max_time):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u, v), max(u, v)))
        
        adj_set = [set(adj[i]) for i in range(n)]
        
        def get_cost():
            return count_short_cycles_heuristic(n, adj, target_girth)
            
        current_cost = get_cost()
        best_cost = current_cost
        best_adj = [list(neighbors) for neighbors in adj]
        best_edge_set = set(edge_set)
        
        temperature = 10.0
        cooling_rate = 0.99
        min_temperature = 0.1
        max_iter = 20000
        
        for i in range(max_iter):
            if time.time() > max_time:
                break
            if current_cost == 0:
                return n, list(edge_set)
                
            # Pick two random edges
            attempts = 0
            while attempts < 10:
                u1 = random.randint(0, n - 1)
                v1 = random.choice(adj[u1])
                u2 = random.randint(0, n - 1)
                v2 = random.choice(adj[u2])
                
                if len({u1, v1, u2, v2}) == 4:
                    key1 = (min(u1, u2), max(u1, u2))
                    key2 = (min(v1, v2), max(v1, v2))
                    if key1 not in edge_set and key2 not in edge_set:
                        break
                attempts += 1
            else:
                continue
                
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            edge_set.remove((min(u1, v1), max(u1, v1)))
            edge_set.remove((min(u2, v2), max(u2, v2)))
            
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            
            edge_set.add(key1)
            edge_set.add(key2)
            
            new_cost = get_cost()
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = [list(neighbors) for neighbors in adj]
                    best_edge_set = set(edge_set)
            else:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                edge_set.remove(key1)
                edge_set.remove(key2)
                
                adj[u1].append(v1)
                adj[v1].append(u1)
                adj[u2].append(v2)
                adj[v2].append(u2)
                
                edge_set.add((min(u1, v1), max(u1, v1)))
                edge_set.add((min(u2, v2), max(u2, v2)))
                
                delta = new_cost - current_cost
                if temperature > min_temperature:
                    if random.random() < math.exp(-delta / temperature):
                        # Re-apply
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        adj[u2].remove(v2)
                        adj[v2].remove(u2)
                        
                        edge_set.remove((min(u1, v1), max(u1, v1)))
                        edge_set.remove((min(u2, v2), max(u2, v2)))
                        
                        adj[u1].append(u2)
                        adj[u2].append(u1)
                        adj[v1].append(v2)
                        adj[v2].append(v1)
                        
                        edge_set.add(key1)
                        edge_set.add(key2)
                        current_cost = new_cost
                
                temperature = max(min_temperature, temperature * cooling_rate)
        
        if best_cost == 0:
            return n, list(best_edge_set)
        return None, None

    # Step 2: Local search from random graphs
    # Try a few sizes
    for n_try in [600, 800, 1000, 1116]:
        if time_left() < 20:
            break
        
        for seed in range(10):
            if time_left() < 10:
                break
            
            result = generate_random_regular(n_try, k, seed=seed + n_try)
            if result is None:
                continue
            n_r, edges_r = result
            
            ls_time_budget = min(10, time_left() - 5)
            if ls_time_budget < 1:
                continue
                
            ls_result = local_search_girth(n_r, edges_r, g, time.time() + ls_time_budget)
            if ls_result[0] is not None:
                n_ls, edges_ls = ls_result
                # Verify
                if verify_girth(n_ls, edges_ls, g):
                    if best_n is None or n_ls < best_n:
                        best_n = n_ls
                        best_edges = edges_ls
                    break  # Found valid graph for this n

    if best_n is None:
        # Fallback
        n_try = 1500
        result = generate_random_regular(n_try, k, seed=42)
        if result:
            n_r, edges_r = result
            if verify_girth(n_r, edges_r, g):
                best_n = n_r
                best_edges = edges_r

    if best_n is None:
        # Emergency fallback
        n_try = 2000
        for seed in range(20):
            result = generate_random_regular(n_try, k, seed=seed)
            if result:
                n_r, edges_r = result
                if verify_girth(n_r, edges_r, g):
                    best_n = n_r
                    best_edges = edges_r
                    break

    return best_n, best_edges
# EVOLVE-BLOCK-END
