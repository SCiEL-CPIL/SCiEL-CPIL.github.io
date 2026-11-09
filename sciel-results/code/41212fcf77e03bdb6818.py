import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set(base_edges)
    
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 != 0 and (u, v) not in base_set:
                candidates.append((u, v))
                
    def get_diameter(added_edges):
        adj = [[] for _ in range(n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_dist = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            for d in dist:
                if d == -1:
                    return float('inf')
                if d > max_dist:
                    max_dist = d
        return max_dist

    best_set = set()
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    def greedy_remove(s):
        s = set(s)
        edges_list = list(s)
        random.shuffle(edges_list)
        for e in edges_list:
            if e in s:
                new_s = s - {e}
                d = get_diameter(new_s)
                if d <= 4:
                    s = new_s
        return s

    def local_search(s, time_limit):
        s = set(s)
        last_improvement = time.time()
        while time.time() < time_limit:
            improved = False
            # Try removals
            edges_list = list(s)
            random.shuffle(edges_list)
            for e in edges_list:
                if e in s:
                    new_s = s - {e}
                    d = get_diameter(new_s)
                    if d <= 4:
                        s = new_s
                        improved = True
                        last_improvement = time.time()
            if improved:
                continue
            
            # Try swaps
            if time.time() - last_improvement > 0.5:
                for _ in range(200):
                    if time.time() > time_limit: break
                    if len(s) == 0: break
                    e_out = random.choice(list(s))
                    e_in = random.choice(candidates)
                    if e_in in s: continue
                    new_s = (s - {e_out}) | {e_in}
                    d = get_diameter(new_s)
                    if d <= 4:
                        s = new_s
                        last_improvement = time.time()
                        break
        return s

    # Initial run: Full set
    current_set = set(candidates)
    current_set = greedy_remove(current_set)
    current_set = local_search(current_set, deadline)
    if len(current_set) < best_size:
        best_set = current_set
        best_size = len(best_set)

    # Additional restarts with heuristic growth
    for _ in range(50):
        if time.time() > deadline: break
        s = set()
        # Add random edges until valid?
        # Better: Start with a random subset of size 20, check if valid.
        # If not, add edges to fix.
        # Actually, let's just use the local search on the best found so far with different random seeds.
        # Or try to start from a smaller random set and grow.
        
        # Heuristic: Start with 10 random edges, try to fix diameter.
        s = set(random.sample(candidates, min(10, len(candidates))))
        d = get_diameter(s)
        while d > 4 and time.time() < deadline:
            # Find edge to add
            best_e = None
            best_d = d
            # Sample 20 candidates
            sample = random.sample(candidates, min(20, len(candidates)))
            for e in sample:
                if e in s: continue
                new_s = s | {e}
                nd = get_diameter(new_s)
                if nd < best_d:
                    best_d = nd
                    best_e = e
            if best_e is None:
                # Pick random
                for e in candidates:
                    if e not in s:
                        best_e = e
                        break
            if best_e is not None:
                s.add(best_e)
                d = best_d
            else:
                break
        
        if d <= 4:
            s = greedy_remove(s)
            s = local_search(s, deadline)
            if len(s) < best_size:
                best_set = s
                best_size = len(best_set)

    result = [[u, v] for u, v in best_set]
    return result
# EVOLVE-BLOCK-END
