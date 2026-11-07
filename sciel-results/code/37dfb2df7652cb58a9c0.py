# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools
import random
import math

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit, keep 10s buffer)
    
    n = P['n']
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        base_edge_set.add(frozenset((u, v)))
    
    # Generate all candidate edges (not in base, and |u-v| != 2 to avoid immediate triangle with path)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if frozenset((u, v)) in base_edge_set:
                continue
            if abs(u - v) == 2:
                continue
            all_candidates.append((u, v))
    
    # Sort candidates by distance (longer first for greedy)
    all_candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    def build_full_adj(added_edges):
        """Build adjacency list from base + added edges."""
        adj = [set(s) for s in base_adj]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        """Check if graph with given adjacency is triangle-free."""
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check if u and v share a common neighbor
                    if adj[u].intersection(adj[v]):
                        return False
        return True
    
    def get_diameter(adj):
        """Compute diameter of graph given adjacency list."""
        max_dist = 0
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        q.append(v)
        return max_dist
    
    def is_valid_solution(added_edges):
        """Check if the solution is valid: triangle-free, connected, diameter <= 4."""
        adj = build_full_adj(added_edges)
        
        # Check triangle-free
        if not is_triangle_free(adj):
            return False
        
        # Check diameter
        diam = get_diameter(adj)
        if diam > 4:
            return False
        
        # Check no duplicate edges or edges in base
        edge_set = set()
        for u, v in added_edges:
            es = frozenset((u, v))
            if es in edge_set or es in base_edge_set:
                return False
            edge_set.add(es)
        
        return True
    
    def get_uncovered_pairs(adj):
        """Get pairs of vertices with distance > 4."""
        uncovered = set()
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for j in range(i + 1, n):
                if dist[j] > 4:
                    uncovered.add((i, j))
        return uncovered
    
    # Greedy construction
    current_adj = [set(s) for s in base_adj]
    chosen_edges = []
    chosen_set = set()
    
    def is_valid_addition(u, v, adj_list):
        """Check if adding (u,v) keeps triangle-free property."""
        # Check if u and v have a common neighbor in current graph
        if adj_list[u].intersection(adj_list[v]):
            return False
        return True
    
    uncovered = get_uncovered_pairs(current_adj)
    
    while uncovered and time.time() < deadline:
        best_edge = None
        best_new_uncovered = None
        max_covered = -1
        
        valid_candidates = []
        for u, v in all_candidates:
            if frozenset((u, v)) in chosen_set:
                continue
            if is_valid_addition(u, v, current_adj):
                valid_candidates.append((u, v))
                
        if not valid_candidates:
            break
        
        for u, v in valid_candidates:
            # Tentatively add
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            new_uncovered = get_uncovered_pairs(current_adj)
            covered_count = len(uncovered) - len(new_uncovered)
            
            # Rollback
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if covered_count > max_covered:
                max_covered = covered_count
                best_edge = (u, v)
                best_new_uncovered = new_uncovered
                
        if max_covered <= 0:
            break
        
        # Commit best edge
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        chosen_edges.append((u, v))
        chosen_set.add(frozenset((u, v)))
        uncovered = best_new_uncovered
    
    # Verify greedy solution
    greedy_solution = list(chosen_edges)
    if is_valid_solution(greedy_solution):
        best_solution = greedy_solution
    else:
        best_solution = None
    
    # Local search: try to remove edges
    if best_solution and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_solution)):
                candidate = best_solution[:i] + best_solution[i+1:]
                if is_valid_solution(candidate):
                    best_solution = candidate
                    improved = True
                    break
    
    # If greedy failed, try a different approach: start from scratch with randomization
    if not best_solution and time.time() < deadline:
        # Try multiple random greedy runs
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.shuffle(all_candidates)
            cur_adj = [set(s) for s in base_adj]
            cur_edges = []
            cur_set = set()
            
            unc = get_uncovered_pairs(cur_adj)
            while unc:
                found = False
                for u, v in all_candidates:
                    if frozenset((u, v)) in cur_set:
                        continue
                    if not is_valid_addition(u, v, cur_adj):
                        continue
                    
                    cur_adj[u].add(v)
                    cur_adj[v].add(u)
                    new_unc = get_uncovered_pairs(cur_adj)
                    if len(new_unc) < len(unc):
                        cur_edges.append((u, v))
                        cur_set.add(frozenset((u, v)))
                        unc = new_unc
                        found = True
                        break
                    
                    cur_adj[u].remove(v)
                    cur_adj[v].remove(u)
                
                if not found:
                    break
            
            if is_valid_solution(cur_edges):
                # Try local search
                sol = list(cur_edges)
                imp = True
                while imp and time.time() < deadline:
                    imp = False
                    for i in range(len(sol)):
                        cand = sol[:i] + sol[i+1:]
                        if is_valid_solution(cand):
                            sol = cand
                            imp = True
                            break
                if best_solution is None or len(sol) < len(best_solution):
                    best_solution = sol
    
    # Final attempt: if still no solution, try to build one systematically
    if not best_solution and time.time() < deadline:
        # Systematic search: try all combinations of small size
        # Start with 5 edges, then 6, etc.
        for k in range(5, 10):
            if time.time() > deadline:
                break
            
            # Use a heuristic: prioritize edges that cover many far-apart pairs
            # Precompute coverage for each candidate
            def coverage_score(u, v):
                """Estimate how many pairs (i,j) with dist(i,j)>4 would be improved by adding (u,v)."""
                # Rough estimate: pairs where u or v is a "hub"
                score = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(i - j) > 4:
                            # Check if adding (u,v) could help reduce dist(i,j)
                            # This is a rough heuristic
                            if (u == i or u == j or v == i or v == j):
                                score += 1
                return score
            
            scored_candidates = [(u, v, coverage_score(u, v)) for u, v in all_candidates]
            scored_candidates.sort(key=lambda x: x[2], reverse=True)
            top_candidates = [x[:2] for x in scored_candidates[:30]]
            
            # Try combinations of k from top_candidates
            for combo in itertools.combinations(top_candidates, k):
                if time.time() > deadline:
                    break
                if is_valid_solution(list(combo)):
                    best_solution = list(combo)
                    break
            if best_solution:
                break
    
    # Return the best solution found
    if best_solution:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Last resort: return empty (invalid, but avoid crash)
    return []

# EVOLVE-BLOCK-END
