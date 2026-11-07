import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    # All valid candidates: opposite parity (guarantees triangle-free with path), not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u + v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
    
    def build_adj(edge_set):
        adj = [[] for _ in range(n)]
        for u, v in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def bfs_diam_and_pairs(edge_set):
        """Return (diameter, list of pairs at max distance)."""
        adj = build_adj(edge_set)
        max_d = 0
        max_pairs = []
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                        q.append(nb)
        # Collect pairs at max distance
        if max_d > 0:
            adj2 = build_adj(edge_set)
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    node = q.popleft()
                    for nb in adj2[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            q.append(nb)
                for v in range(src + 1, n):
                    if dist[v] == max_d:
                        max_pairs.append((src, v))
        return max_d, max_pairs
    
    def check_diam_leq4(edge_set):
        """Quick check if diameter <= 4 using BFS."""
        adj = build_adj(edge_set)
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                if dist[node] > 4:
                    return False
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
        return True
    
    def greedy_build():
        """Greedy: repeatedly add the edge that most reduces diameter."""
        current_edges = set(base_edges)
        added = []
        
        for _ in range(30):
            if time.time() > deadline:
                break
            
            diam, max_pairs = bfs_diam_and_pairs(current_edges)
            if diam <= 4:
                break
            
            # Score each candidate by how many max-distance pairs it reduces
            best_cand = None
            best_score = -1
            
            for u, v in candidates:
                if (u, v) in current_edges:
                    continue
                
                new_edges = current_edges | {(u, v)}
                # Quick check: does this help?
                adj = build_adj(new_edges)
                
                # Check if diameter improves
                new_diam = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    q = deque([src])
                    while q:
                        node = q.popleft()
                        for nb in adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                if dist[nb] > new_diam:
                                    new_diam = dist[nb]
                                q.append(nb)
                
                if new_diam >= diam:
                    continue
                
                # Count how many max_pairs are resolved
                score = 0
                for p1, p2 in max_pairs:
                    # Check if dist(p1, p2) < diam in new graph
                    dist = [-1] * n
                    dist[p1] = 0
                    q = deque([p1])
                    while q:
                        node = q.popleft()
                        for nb in adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                q.append(nb)
                    if dist[p2] < diam:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_cand = (u, v)
            
            if best_cand is None:
                # Try any edge that reduces diameter
                for u, v in candidates:
                    if (u, v) in current_edges:
                        continue
                    new_edges = current_edges | {(u, v)}
                    adj = build_adj(new_edges)
                    new_diam = 0
                    for src in range(n):
                        dist = [-1] * n
                        dist[src] = 0
                        q = deque([src])
                        while q:
                            node = q.popleft()
                            for nb in adj[node]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[node] + 1
                                    if dist[nb] > new_diam:
                                        new_diam = dist[nb]
                                    q.append(nb)
                    if new_diam < diam:
                        best_cand = (u, v)
                        break
            
            if best_cand is None:
                break
            
            current_edges.add(best_cand)
            added.append(list(best_cand))
        
        return added
    
    # Try greedy first
    best_added = greedy_build()
    best_len = len(best_added)
    
    # Try to improve by removing edges (local search)
    if best_len > 0 and time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i in range(len(best_added)):
                # Try removing best_added[i]
                test_added = best_added[:i] + best_added[i+1:]
                test_edges = set(base_edges)
                for e in test_added:
                    test_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                
                if check_diam_leq4(test_edges):
                    best_added = test_added
                    best_len = len(best_added)
                    improved = True
                    break
            
            if not improved:
                break
    
    # Random restarts to try to find better solutions
    if time.time() < deadline - 10.0:
        for attempt in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Randomized greedy
            current_edges = set(base_edges)
            added = []
            
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                
                diam, max_pairs = bfs_diam_and_pairs(current_edges)
                if diam <= 4:
                    break
                
                # Score candidates with some randomness
                scored = []
                for u, v in candidates:
                    if (u, v) in current_edges:
                        continue
                    
                    new_edges = current_edges | {(u, v)}
                    adj = build_adj(new_edges)
                    
                    new_diam = 0
                    for src in range(n):
                        dist = [-1] * n
                        dist[src] = 0
                        q = deque([src])
                        while q:
                            node = q.popleft()
                            for nb in adj[node]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[node] + 1
                                    if dist[nb] > new_diam:
                                        new_diam = dist[nb]
                                    q.append(nb)
                    
                    if new_diam >= diam:
                        continue
                    
                    # Score: how many max pairs resolved
                    score = 0
                    for p1, p2 in max_pairs:
                        dist = [-1] * n
                        dist[p1] = 0
                        q = deque([p1])
                        while q:
                            node = q.popleft()
                            for nb in adj[node]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[node] + 1
                                    q.append(nb)
                        if dist[p2] < diam:
                            score += 1
                    
                    # Add randomness
                    score += random.random() * 0.5
                    scored.append((score, (u, v)))
                
                if not scored:
                    break
                
                scored.sort(key=lambda x: -x[0])
                best_cand = scored[0][1]
                
                current_edges.add(best_cand)
                added.append(list(best_cand))
            
            if len(added) < best_len:
                best_added = added
                best_len = len(added)
                
                # Try to remove more
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    improved = False
                    for i in range(len(best_added)):
                        test_added = best_added[:i] + best_added[i+1:]
                        test_edges = set(base_edges)
                        for e in test_added:
                            test_edges.add((min(e[0], e[1]), max(e[0], e[1])))
                        if check_diam_leq4(test_edges):
                            best_added = test_added
                            best_len = len(best_added)
                            improved = True
                            break
                    if not improved:
                        break
    
    # Ensure proper types
    result = []
    for e in best_added:
        u = int(e[0])
        v = int(e[1])
        if u > v:
            u, v = v, u
        result.append([u, v])
    
    return result
