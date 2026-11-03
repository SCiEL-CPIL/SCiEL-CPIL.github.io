import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_c, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_c]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    seed_n, seed_edges = luw_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Non-backtracking walk girth check from every vertex."""
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        return False
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, d + 1))
        return True

    def count_short_cycles(n, adj, g):
        """Count number of short cycles using non-backtracking walks."""
        count = 0
        for s in range(n):
            visited = set()
            visited.add((s, -1))
            queue = deque([(s, -1, 0)])
            while queue:
                u, prev, d = queue.popleft()
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s and d + 1 < g:
                        count += 1
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((v, u, d + 1))
        return count

    def generate_simple_regular(n, k, max_attempts=50):
        """Generate a simple k-regular graph using configuration model with repair."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Create stubs and pair
            stubs = list(range(n)) * k
            random.shuffle(stubs)
            edges = []
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u != v:
                    edges.append((u, v))
            
            # Build edge dict for duplicate tracking
            edge_dict = {}
            for i, (u, v) in enumerate(edges):
                key = (min(u, v), max(u, v))
                if key not in edge_dict:
                    edge_dict[key] = []
                edge_dict[key].append(i)
            
            # Check if already simple
            is_simple = all(len(v) == 1 for v in edge_dict.values())
            
            if is_simple and len(edges) == n * k // 2:
                return edges
            
            # Repair with 2-switches
            max_swaps = min(n * k, 2000)
            for swap_iter in range(max_swaps):
                if time.time() > deadline - 5:
                    return None
                
                # Find a duplicate edge
                bad_key = None
                bad_idx1 = None
                for key, indices in edge_dict.items():
                    if len(indices) > 1:
                        bad_key = key
                        bad_idx1 = indices[0]
                        break
                
                if bad_key is None:
                    break
                
                u1, v1 = edges[bad_idx1]
                
                # Try to swap with another edge
                found_swap = False
                for j in random.sample(range(len(edges)), min(50, len(edges))):
                    if j == bad_idx1:
                        continue
                    u2, v2 = edges[j]
                    
                    # Try swapping: (u1,v1) and (u2,v2) -> (u1,v2) and (u2,v1)
                    new_u1, new_v1 = u1, v2
                    new_u2, new_v2 = u2, v1
                    
                    if new_u1 == new_v1 or new_u2 == new_v2:
                        continue
                    
                    key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
                    key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
                    
                    if key1 == bad_key and key2 == bad_key:
                        continue
                    
                    old_key1 = (min(u1, v1), max(u1, v1))
                    old_key2 = (min(u2, v2), max(u2, v2))
                    
                    # Check if this would create new duplicates
                    if (key1 in edge_dict and len(edge_dict[key1]) >= 1 and key1 != old_key1) or \
                       (key2 in edge_dict and len(edge_dict[key2]) >= 1 and key2 != old_key2):
                        continue
                    
                    # Apply swap
                    edges[bad_idx1] = (new_u1, new_v1)
                    edges[j] = (new_u2, new_v2)
                    
                    # Update edge_dict
                    if old_key1 in edge_dict:
                        edge_dict[old_key1].remove(bad_idx1)
                        if not edge_dict[old_key1]:
                            del edge_dict[old_key1]
                    if old_key2 in edge_dict:
                        edge_dict[old_key2].remove(j)
                        if not edge_dict[old_key2]:
                            del edge_dict[old_key2]
                    
                    if key1 not in edge_dict:
                        edge_dict[key1] = []
                    edge_dict[key1].append(bad_idx1)
                    
                    if key2 not in edge_dict:
                        edge_dict[key2] = []
                    edge_dict[key2].append(j)
                    
                    found_swap = True
                    break
                
                if not found_swap:
                    break
            
            # Final check
            edge_set = set()
            is_simple = True
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    is_simple = False
                    break
                edge_set.add(key)
            
            if is_simple and len(edges) == n * k // 2:
                return edges
        
        return None

    def local_search_reduce_cycles(n, adj, edges, g, max_iters=500):
        """Local search to reduce number of short cycles."""
        best_edges = list(edges)
        best_count = count_short_cycles(n, adj, g)
        
        if best_count == 0:
            return best_edges, 0, adj
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            if best_count == 0:
                break
            
            # Try multiple swaps per iteration
            improved = False
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random edge
                i = random.randint(0, len(best_edges) - 1)
                u1, v1 = best_edges[i]
                
                # Try swapping with a random other edge
                j = random.randint(0, len(best_edges) - 1)
                if j == i:
                    continue
                u2, v2 = best_edges[j]
                
                # Try swap: (u1,v1) and (u2,v2) -> (u1,v2) and (u2,v1)
                new_u1, new_v1 = u1, v2
                new_u2, new_v2 = u2, v1
                
                if new_u1 == new_v1 or new_u2 == new_v2:
                    continue
                
                # Check for duplicate edges
                key1 = (min(new_u1, new_v1), max(new_u1, new_v1))
                key2 = (min(new_u2, new_v2), max(new_u2, new_v2))
                
                # Quick check using edge set
                edge_set = set()
                for idx, (u, v) in enumerate(best_edges):
                    if idx != i and idx != j:
                        edge_set.add((min(u, v), max(u, v)))
                
                if key1 in edge_set or key2 in edge_set or key1 == key2:
                    continue
                
                # Apply swap and check
                best_edges[i] = (new_u1, new_v1)
                best_edges[j] = (new_u2, new_v2)
                
                new_adj = build_adj(best_edges, n)
                new_count = count_short_cycles(n, new_adj, g)
                
                if new_count < best_count:
                    best_count = new_count
                    adj = new_adj
                    improved = True
                    break
                else:
                    # Revert
                    best_edges[i] = (u1, v1)
                    best_edges[j] = (u2, v2)
            
            if not improved and it > 100:
                break
        
        return best_edges, best_count, adj

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # For k=5, g=9: Moore bound is 428, seed is 1116
    # Try to find smaller graphs with girth >= 9
    
    # Strategy: try n values from just above Moore bound up to seed_n
    search_start = max(moore_bound, 430)
    if search_start % 2 != 0:
        search_start += 1
    search_end = min(seed_n, 700)
    if search_end % 2 != 0:
        search_end -= 1
    
    if search_start <= search_end:
        # Try n values, starting from smaller n
        n_values = list(range(search_start, search_end + 1, 2))
        
        # Limit total number of n values to try based on time
        max_n_values = 30
        
        for n in n_values[:max_n_values]:
            if time.time() > deadline - 15:
                break
            
            # Try up to 3 random graphs for this n
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                
                edges = generate_simple_regular(n, k, max_attempts=20)
                if edges is None:
                    continue
                
                adj = build_adj(edges, n)
                
                # Check girth
                if exact_girth_check(n, adj, g):
                    return n, edges
                
                # Local search to reduce cycles
                edges, count, adj = local_search_reduce_cycles(n, adj, edges, g, max_iters=300)
                
                if count == 0:
                    if exact_girth_check(n, adj, g):
                        return n, edges
    
    return seed_n, seed_edges
