import time
import random
from collections import defaultdict

def _min_vertex_cover(adj, deadline):
    n = len(adj)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if not edges:
        return []

    def greedy_vertex_cover():
        remaining_edges = set(edges)
        cover = set()
        while remaining_edges:
            max_deg = -1
            best_v = -1
            for u in range(n):
                if u in cover: continue
                cnt = 0
                for v in adj[u]:
                    if (min(u, v), max(u, v)) in remaining_edges:
                        cnt += 1
                if cnt > max_deg:
                    max_deg = cnt
                    best_v = u
            if best_v == -1: break
            cover.add(best_v)
            remaining_edges = {e for e in remaining_edges if best_v not in e}
        return cover
    
    best_cover = greedy_vertex_cover()
    best_size = len(best_cover)
    
    def solve(remaining_edges, current_cover):
        nonlocal best_cover, best_size
        
        if time.time() > deadline:
            return
        
        if not remaining_edges:
            if len(current_cover) < best_size:
                best_size = len(current_cover)
                best_cover = set(current_cover)
            return
            
        # Lower bound: Greedy matching
        matched = set()
        lb = 0
        for u, v in remaining_edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                lb += 1
                
        if len(current_cover) + lb >= best_size:
            return

        # Pick edge with max degree sum
        deg_rem = defaultdict(int)
        for u, v in remaining_edges:
            deg_rem[u] += 1
            deg_rem[v] += 1
            
        max_score = -1
        branch_edge = None
        for u, v in remaining_edges:
            score = deg_rem[u] + deg_rem[v]
            if score > max_score:
                max_score = score
                branch_edge = (u, v)
                
        if not branch_edge:
            return
            
        u, v = branch_edge
        
        # Branch 1: Add u
        new_cover_1 = current_cover | {u}
        new_remaining_1 = set()
        for e in remaining_edges:
            if u not in e:
                new_remaining_1.add(e)
        solve(new_remaining_1, new_cover_1)
        
        # Branch 2: Add v
        new_cover_2 = current_cover | {v}
        new_remaining_2 = set()
        for e in remaining_edges:
            if v not in e:
                new_remaining_2.add(e)
        solve(new_remaining_2, new_cover_2)

    solve(set(edges), set())
    return sorted(best_cover)

def construct():
    start_total = time.time()
    limit = 115.0
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        elapsed = time.time() - start_total
        remaining_time = limit - elapsed
        
        if remaining_time < 10:
            def greedy_vc(adj):
                n = len(adj)
                edges = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v: edges.add((u, v))
                cover = set()
                while edges:
                    max_deg = -1
                    best_v = -1
                    for u in range(n):
                        if u in cover: continue
                        cnt = sum(1 for v in adj[u] if (min(u,v), max(u,v)) in edges)
                        if cnt > max_deg:
                            max_deg = cnt
                            best_v = u
                    if best_v == -1: break
                    cover.add(best_v)
                    edges = {e for e in edges if best_v not in e}
                return sorted(cover)
            results.append(greedy_vc(adj))
        else:
            instance_deadline = time.time() + remaining_time * 0.8
            results.append(_min_vertex_cover(adj, instance_deadline))
            
    return results
# EVOLVE-BLOCK-END
