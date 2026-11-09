import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _find_2factor_cycles(adj, n):
        """
        Find vertex-disjoint cycles via bipartite matching (2-factor extraction).
        Returns list of cycles found.
        """
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        
        m = len(edge_list)
        if m == 0:
            return []
        
        INF = 10**9
        cost = [[INF] * (2 * n) for _ in range(m)]
        
        for i, (u, v) in enumerate(edge_list):
            cost[i][2 * u] = 0
            cost[i][2 * u + 1] = 0
            cost[i][2 * v] = 0
            cost[i][2 * v + 1] = 0
            
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            return []
        
        matched_edges = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i][j] < INF:
                matched_edges.add(edge_list[i])
        
        if not matched_edges:
            return []
            
        sub_adj = {v: set() for v in range(n)}
        for u, v in matched_edges:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
            
        cycles = []
        visited = set()
        
        for start in range(n):
            if start in visited or len(sub_adj[start]) != 2:
                continue
                
            path = [start]
            visited.add(start)
            prev = -1
            curr = start
            
            while True:
                neighbors = list(sub_adj[curr])
                if not neighbors:
                    break
                    
                if prev == -1:
                    nxt = neighbors[0]
                else:
                    if len(neighbors) == 2:
                        if neighbors[0] == prev:
                            nxt = neighbors[1]
                        else:
                            nxt = neighbors[0]
                    else:
                        nxt = neighbors[0]
                        
                if nxt == start:
                    if len(path) >= 3:
                        cycles.append(path)
                    break
                elif nxt in visited:
                    break
                else:
                    prev = curr
                    curr = nxt
                    path.append(curr)
                    visited.add(curr)
                    
        return cycles

    def _find_cycle_dfs(adj, start, rng, max_attempts=30):
        """Find a cycle through start using randomized iterative DFS."""
        best_cycle = None
        best_len = 0
        
        for attempt in range(max_attempts):
            visited = {start}
            path = [start]
            stack = [(start, list(adj[start]))]
            found = False
            
            while stack and not found:
                node, neighbors = stack[-1]
                if not neighbors:
                    stack.pop()
                    path.pop()
                    continue
                nxt = neighbors.pop()
                if nxt == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = list(path)
                    found = True
                    break
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    stack.append((nxt, list(adj[nxt])))
            
            if best_len >= 12:
                break
        
        return best_cycle

    def decompose(edges, n):
        if not edges:
            return []
        
        best_pieces = None
        best_count = float('inf')
        
        num_restarts = 80
        
        for r in range(num_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(42 + r)
            current_edges = set(edges)
            adj = defaultdict(set)
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            pieces = []
            
            # Phase 1: 2-factor extraction (multiple rounds)
            for _ in range(8):
                if time.time() - start_time > DEADLINE:
                    break
                if not current_edges:
                    break
                if len(current_edges) < 6:
                    break
                
                f_cycles = _find_2factor_cycles(adj, n)
                
                if not f_cycles:
                    break
                    
                added_any = False
                for c in f_cycles:
                    valid = True
                    for i in range(len(c)):
                        u, v = c[i], c[(i + 1) % len(c)]
                        e = (min(u, v), max(u, v))
                        if e not in current_edges:
                            valid = False
                            break
                    if valid:
                        pieces.append(list(c))
                        for i in range(len(c)):
                            u, v = c[i], c[(i + 1) % len(c)]
                            e = (min(u, v), max(u, v))
                            current_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                        added_any = True
                if not added_any:
                    break

            # Phase 2: Greedy DFS for remaining edges
            while current_edges:
                if time.time() - start_time > DEADLINE:
                    break
                    
                # Find vertices with degree >= 2
                vertices = [u for u in adj if len(adj[u]) >= 2]
                if not vertices:
                    break
                    
                # Try multiple random starts to find the longest cycle
                best_cycle = None
                rng.shuffle(vertices)
                for start in vertices[:15]:
                    c = _find_cycle_dfs(adj, start, rng, max_attempts=20)
                    if c and (best_cycle is None or len(c) > len(best_cycle)):
                        best_cycle = c
                        if len(best_cycle) >= 15:
                            break
                
                if best_cycle is None:
                    break
                
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    current_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                
                pieces.append(list(best_cycle))
            
            # Add remaining edges as single pieces
            for e in current_edges:
                pieces.append([e[0], e[1]])
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        n = spec[0]
        results.append(decompose(edges, n))
    return results

if __name__ == "__main__":
    construct()
