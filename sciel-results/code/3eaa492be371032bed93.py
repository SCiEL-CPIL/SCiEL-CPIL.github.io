import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Baseline: LUW Construction (Guaranteed valid)
    # We use d=11 for k=3, g=16 to ensure girth >= 16.
    # n = 2 * (3^11 - 1) / 2 = 177147
    # This is a valid 3-regular graph with girth 2*11+1 = 23 >= 16.
    # It serves as the fallback if the voltage lift fails or is larger.
    
    q = k
    d = 11
    # LUW rules for k=3 (Z_3)
    # We hardcode the known LUW construction for k=3, d=11
    # Actually, let's just use the standard LUW algorithm from the seed but with d=11.
    # The seed used d=11 implicitly? No, seed used d=11 logic? 
    # Seed: d = max(3, g-5 if (g-5)%2 else g-4). For g=16, d=11.
    # So the seed already generates the LUW graph of size 177147.
    # We will reproduce this exactly to have a valid baseline.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours_luw(vertex):
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
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    baseline_n = len(order)
    baseline_edges = edges
    
    best_n = baseline_n
    best_edges = baseline_edges
    
    # 2. Voltage Lift Strategy
    # Base: K3,3 (6 vertices, 3-regular, girth 4)
    # Lift over Z_m. Total vertices N = 6 * m.
    # We want girth >= 16.
    # Moore bound for k=3, g=16 is 512.
    # If m=100, N=600.
    # If m=200, N=1200.
    # Let's try m values that result in N > Moore bound.
    
    # K3,3 structure:
    # Part U: 0, 1, 2
    # Part V: 3, 4, 5
    # Edges: (0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)
    
    # Voltage assignment: Each edge e=(u,v) has a voltage gamma(e) in Z_m.
    # If we traverse u->v, we add gamma(u,v). v->u subtracts gamma(u,v) (or adds -gamma).
    # For undirected graph, we assign voltages to directed edges such that gamma(u,v) = -gamma(v,u).
    # We can represent voltages for the 9 edges.
    
    # The girth of the lift is determined by the shortest non-backtracking walk in the base graph
    # such that the sum of voltages is 0 mod m.
    
    # Base graph non-backtracking walks of length L:
    # Length 4: 0-3-4-0-? No, 0-3-4-0 is a cycle in base.
    # We need to check walks in the base graph that close up in the lift.
    
    # Let's define the base graph adjacency with edge indices.
    # Base edges:
    # e0: 0-3
    # e1: 0-4
    # e2: 0-5
    # e3: 1-3
    # e4: 1-4
    # e5: 1-5
    # e6: 2-3
    # e7: 2-4
    # e8: 2-5
    
    # To have girth >= 16, we need no closed non-backtracking walk of length < 16 in the lift.
    # In the base graph, shortest cycles are length 4.
    # A lift cycle of length L corresponds to a walk of length L in the base graph
    # that is non-backtracking and has total voltage 0.
    
    # We can simulate annealing on the voltages v0..v8 (integers mod m).
    # Objective: Maximize the girth.
    # Heuristic for girth: 
    # 1. Check all base cycles of length 4, 6, 8, ... up to 15.
    # 2. For each cycle, compute the set of voltage sums.
    # 3. If 0 is in the set, girth is <= L.
    # 4. We want to avoid 0 for L < 16.
    
    # However, checking all walks is expensive.
    # Alternative: Randomize voltages and check girth via BFS on the lift graph.
    # N=600 is small enough for BFS to be fast.
    
    def build_lift(m, voltages):
        # voltages: list of 9 ints
        N = 6 * m
        edges = set()
        # Base edges: (u, v, idx)
        base_edges = [
            (0, 3, 0), (0, 4, 1), (0, 5, 2),
            (1, 3, 3), (1, 4, 4), (1, 5, 5),
            (2, 3, 6), (2, 4, 7), (2, 5, 8)
        ]
        
        # Vertex in lift: (base_vertex, offset)
        # Edge (u, off_u) to (v, off_v) if off_v = off_u + volt[idx] (for u->v direction in base)
        # Wait, direction matters.
        # Let's define voltages for u->v where u < v in base? No, just assign to each directed edge.
        # Symmetry: gamma(v,u) = -gamma(u,v).
        
        for u, v, idx in base_edges:
            vu = voltages[idx]
            vv = -vu
            # u -> v
            for off in range(m):
                u_lift = u * m + off
                v_lift = v * m + (off + vu) % m
                edges.add((u_lift, v_lift))
            # v -> u
            for off in range(m):
                v_lift = v * m + off
                u_lift = u * m + (off + vv) % m
                edges.add((v_lift, u_lift))
        
        return list(edges), N

    def get_girth_fast(edges, N, target_girth):
        # BFS from each node? No, just check if any cycle < target_girth exists.
        # We can do BFS from a random subset or all nodes if N is small.
        # N=600, BFS from all nodes is 600 * 600 = 360k ops. Very fast.
        
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # If u and v are at same level or adjacent levels?
                        # In BFS, if v is visited and v != parent[u], it's a back/cross edge.
                        # Cycle length is dist[u] + dist[v] + 1 ONLY if v is ancestor of u or vice versa?
                        # No, standard BFS cycle detection for girth:
                        # If we find an edge (u,v) where dist[u] == dist[v] or |dist[u]-dist[v]|==1?
                        # Actually, the shortest cycle through start is found when we close the loop.
                        # But we want the global girth.
                        # A more robust way for girth:
                        # For each edge (u,v), if dist[u] + dist[v] + 1 < current_min and path doesn't overlap improperly?
                        # Simpler: Just check if we can find a cycle of length < target.
                        
                        # Let's just compute the cycle length properly.
                        # If v is not parent of u, and u is not parent of v:
                        # If dist[u] == dist[v]: cycle length = 2*dist[u] + 2? No.
                        # If dist[v] == dist[u] + 1: v is child? No, dist[v] > dist[u] means v is deeper.
                        # If dist[u] > dist[v]: u is deeper.
                        
                        # Let's use a standard algorithm for girth:
                        # For each node, BFS. When we encounter a visited node v that is not parent:
                        # If dist[v] < dist[u], we found a cycle.
                        # Length = dist[u] + dist[v] + 1 is WRONG if v is not ancestor.
                        
                        # Correct approach for girth in unweighted graph:
                        # For each edge (u,v), remove it, find dist(u,v). Girth = min(dist(u,v)+1).
                        # This is O(E*(V+E)). 9*600 * 600 = 3.2M. Acceptable.
                        # But we are doing this inside a loop.
                        
                        # Let's stick to the heuristic: if we find ANY back edge, we have a cycle.
                        # To be safe and fast, let's just return a score based on "did we find a cycle < target".
                        # If we find a cycle < target, return that length.
                        
                        # Reconstruct path to find length? Too slow.
                        # Let's just use the O(E(V+E)) method outside the inner loop, or simplify.
                        
                        # Actually, for the search, we just need to know if girth >= 16.
                        # We can stop BFS if dist[u] >= target_girth // 2?
                        # If we haven't found a cycle by depth D, and D is large enough, we might be safe.
                        
                        # Let's just do the full check but break early if min_girth < target.
                        pass
        
        # The BFS logic above is getting messy. Let's use the Edge-Removal method for accuracy in the final check,
        # but for the SA loop, we can use a faster heuristic or just random restarts with full check.
        
        # Given N=600, full girth check via edge removal:
        # 1800 edges. For each, BFS on 600 nodes. 1800 * 600 = 1M operations.
        # This is very fast in Python if optimized? Maybe 0.1s per check.
        # We can do ~1000 checks in 100s.
        
        # Let's implement a faster girth check:
        # BFS from every node, but stop expanding if dist > target_girth/2.
        # If we find a back edge (u,v) where dist[u] + dist[v] + 1 < target, we found a short cycle.
        # Wait, dist[u] + dist[v] + 1 is the length of the cycle formed by the tree paths from start to u and v, plus the edge (u,v).
        # This is a valid cycle.
        # So if we find ANY such edge, girth <= dist[u] + dist[v] + 1.
        
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= target_girth: # No need to go deeper
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle involving start
                        # Length = dist[u] + dist[v] + 1
                        # This assumes the paths start->u and start->v are disjoint except at start.
                        # They are, because it's a tree.
                        # So this is a valid simple cycle.
                        l = dist[u] + dist[v] + 1
                        if l < min_girth:
                            min_girth = l
                            if min_girth < target_girth:
                                return min_girth
        
        return min_girth

    # Simulated Annealing / Random Search on Voltages
    # We try different m values.
    # m=100 -> N=600.
    # m=150 -> N=900.
    # m=200 -> N=1200.
    
    candidate_ms = [100, 120, 150, 200, 300]
    
    for m in candidate_ms:
        if time.time() > deadline - 10.0:
            break
            
        N = 6 * m
        if N >= best_n:
            continue # No point trying if already larger than best
        
        # Initialize voltages randomly
        voltages = [random.randint(0, m-1) for _ in range(9)]
        
        # Evaluate
        edges, _ = build_lift(m, voltages)
        girth = get_girth_fast(edges, N, g)
        
        if girth >= g:
            best_n = N
            best_edges = edges
            break # Found a valid one with this m. Can we do smaller m?
            # We are iterating m in increasing order, so this is the smallest m found so far.
        
        # Try to improve girth via local search
        # Pick a random voltage and change it.
        # If girth increases, accept.
        
        current_girth = girth
        for step in range(1000):
            if time.time() > deadline - 10.0:
                break
            
            # Pick random edge index
            idx = random.randint(0, 8)
            old_v = voltages[idx]
            new_v = random.randint(0, m-1)
            if new_v == old_v:
                continue
            voltages[idx] = new_v
            
            edges, _ = build_lift(m, voltages)
            new_girth = get_girth_fast(edges, N, g)
            
            if new_girth >= g:
                best_n = N
                best_edges = edges
                break # Found valid graph for this m
            
            # Accept if better girth (greedy) or with probability (annealing)
            if new_girth > current_girth:
                current_girth = new_girth
            else:
                # Revert
                voltages[idx] = old_v
        
        if best_n < baseline_n:
            break # Found a valid graph smaller than baseline. Stop.

    return best_n, best_edges
