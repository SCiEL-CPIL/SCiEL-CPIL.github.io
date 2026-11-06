"""Find longest simple path in pinned G(100, 1.8/100) graph using branch-and-bound DFS."""
import time
import random
import sys

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = P['n']
    
    # Build adjacency as bitmasks for fast set operations
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                m = adj_mask[v]
                while m:
                    lb = m & (-m)
                    nb = lb.bit_length() - 1
                    m ^= lb
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    if not components:
        return [0]
    
    # Sort by size, largest first
    components.sort(key=len, reverse=True)
    largest = components[0]
    largest_set = set(largest)
    comp_size = len(largest)
    
    # Build component-local adjacency (only edges within largest component)
    comp_adj = [0] * n  # bitmask of neighbors within the component
    for v in largest:
        comp_adj[v] = adj_mask[v]
    
    # Degree within component
    comp_deg = [bin(comp_adj[v]).count('1') for v in largest]
    
    best_path = []
    best_len = 0
    
    # Greedy initialization: try Warnsdorff-style from several starts
    def greedy_from(start):
        path = [start]
        used = 1 << start
        while True:
            cur = path[-1]
            cands = comp_adj[cur] & ~used
            if not cands:
                break
            # Pick neighbor with fewest unvisited neighbors (Warnsdorff)
            best_v = -1
            best_score = 999
            m = cands
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                score = bin(comp_adj[v] & ~used).count('1')
                if score < best_score or (score == best_score and v < best_v):
                    best_score = score
                    best_v = v
            path.append(best_v)
            used |= (1 << best_v)
        return path
    
    # Try greedy from multiple starts
    random.seed(42)
    # Try all vertices with degree 1 (leaves) first, then others
    starts = sorted(largest, key=lambda v: (comp_deg[v], v))
    for s in starts[:min(20, comp_size)]:
        p = greedy_from(s)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Also try from random starts
    for _ in range(20):
        s = random.choice(largest)
        p = greedy_from(s)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # If we already have a Hamiltonian path, we're done
    if best_len == comp_size:
        return best_path
    
    # Branch and bound DFS
    # We'll search for a longer path using the best found as lower bound
    def branch_and_bound(start, time_budget):
        nonlocal best_path, best_len
        t_start = time.time()
        
        path = [start]
        used = 1 << start
        path_len = 1
        
        # Count of unvisited vertices reachable from current endpoint
        # (upper bound: path_len + reachable_unvisited)
        
        def dfs():
            nonlocal best_path, best_len
            if time.time() - t_start > time_budget:
                return
            if path_len > best_len:
                best_len = path_len
                best_path = list(path)
                if best_len == comp_size:
                    return
            
            cur = path[-1]
            cands = comp_adj[cur] & ~used
            
            if not cands:
                return
            
            # Upper bound: can we possibly beat best_len?
            # Max additional vertices = number of unvisited vertices in component
            # Simple bound: path_len + (comp_size - path_len) = comp_size
            # Tighter: only count vertices reachable from cur through unvisited
            # But that's expensive. Use: if path_len + (comp_size - path_len) <= best_len, prune
            # This is always true, so not useful. Instead:
            # Count unvisited neighbors of cur that lead to unvisited territory
            # Actually, the key pruning: if even visiting ALL remaining vertices can't beat best_len
            # path_len + (comp_size - path_len) = comp_size, so only useful if comp_size <= best_len
            
            if comp_size <= best_len:
                return
            
            # Order candidates by Warnsdorff: fewest onward options first
            cand_list = []
            m = cands
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                onward = bin(comp_adj[v] & ~used).count('1')
                cand_list.append((onward, v))
            
            cand_list.sort()
            
            # Limit branching for performance
            max_branch = 6
            if len(cand_list) > max_branch:
                # Take lowest-degree ones and some random
                chosen = cand_list[:max_branch//2]
                remaining = cand_list[max_branch//2:]
                random.shuffle(remaining)
                chosen.extend(remaining[:max_branch - len(chosen)])
                cand_list = chosen
            
            for onward, v in cand_list:
                if time.time() - t_start > time_budget:
                    return
                # Pruning: if this vertex's onward connections + path_len + 1 <= best_len
                # and we can't reach more than that, skip
                # Actually: after going to v, max path = path_len + 1 + (unvisited reachable from v)
                # Upper bound on reachable: comp_size - path_len - 1 (all remaining)
                # So max possible = path_len + 1 + (comp_size - path_len - 1) = comp_size
                # Only useful if comp_size <= best_len, already checked above.
                
                path.append(v)
                used |= (1 << v)
                path_len += 1
                dfs()
                path_len -= 1
                used ^= (1 << v)
                path.pop()
                
                if best_len == comp_size:
                    return
        
        dfs()
    
    # Run branch and bound from promising starts
    # Priority: leaves first (degree 1), then degree 2, then others
    start_order = sorted(largest, key=lambda v: (comp_deg[v], v))
    
    # Give each start a time slice
    num_starts = min(30, comp_size)
    total_budget = deadline - time.time()
    if total_budget <= 0:
        return best_path if best_path else [0]
    
    per_start = total_budget / num_starts
    
    for i, s in enumerate(start_order[:num_starts]):
        if time.time() > deadline - 2.0:
            break
        if best_len == comp_size:
            break
        remaining = deadline - time.time() - 2.0
        if remaining <= 0:
            break
        t_budget = min(per_start, remaining)
        branch_and_bound(s, t_budget)
    
    # Additional passes with different random seeds for variety
    if best_len < comp_size and time.time() < deadline - 3.0:
        random.seed(12345)
        remaining = deadline - time.time() - 3.0
        if remaining > 0:
            # Try a few more random starts with slightly longer budgets
            for _ in range(5):
                if time.time() > deadline - 2.0:
                    break
                if best_len == comp_size:
                    break
                s = random.choice(largest)
                t_budget = min(2.0, deadline - time.time() - 2.0)
                if t_budget > 0.1:
                    branch_and_bound(s, t_budget)
    
    # Final extension attempt
    if best_path and best_len < comp_size and time.time() < deadline - 1.0:
        seen = set(best_path)
        # Try to extend from both ends
        extended = True
        while extended and time.time() < deadline - 1.0:
            extended = False
            # Front
            if best_path:
                cur = best_path[0]
                cands = comp_adj[cur] & ~(sum(1 << v for v in seen))
                if cands:
                    # Pick one with fewest onward
                    m = cands
                    best_v = -1
                    best_score = 999
                    while m:
                        lb = m & (-m)
                        v = lb.bit_length() - 1
                        m ^= lb
                        score = bin(comp_adj[v] & ~(sum(1 << x for x in seen))).count('1')
                        if score < best_score:
                            best_score = score
                            best_v = v
                    if best_v >= 0:
                        best_path.insert(0, best_v)
                        seen.add(best_v)
                        extended = True
            # Back
            if best_path:
                cur = best_path[-1]
                cands = comp_adj[cur] & ~(sum(1 << v for v in seen))
                if cands:
                    m = cands
                    best_v = -1
                    best_score = 999
                    while m:
                        lb = m & (-m)
                        v = lb.bit_length() - 1
                        m ^= lb
                        score = bin(comp_adj[v] & ~(sum(1 << x for x in seen))).count('1')
                        if score < best_score:
                            best_score = score
                            best_v = v
                    if best_v >= 0:
                        best_path.append(best_v)
                        seen.add(best_v)
                        extended = True
        
        best_len = len(best_path)
    
    if not best_path:
        best_path = [largest[0]]
    
    return best_path
