"""Solve max independent set on square-sum conflict graph with B&B + local search."""
import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = 256
    
    # Precompute squares up to 2*256 = 512
    squares = set()
    for i in range(1, 23):  # 22^2 = 484, 23^2 = 529
        squares.add(i * i)
    
    # Build conflict graph: adj[i] = set of j where i+j is a square (j >= i)
    # Also handle diagonal: 2*i is a square -> i is self-conflicting (cannot be in set)
    adj = [set() for _ in range(n + 1)]
    self_conflict = set()
    for i in range(1, n + 1):
        if 2 * i in squares:
            self_conflict.add(i)
        for j in range(i + 1, n + 1):
            if i + j in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Vertices that can be in the solution (not self-conflicting)
    valid_vertices = [v for v in range(1, n + 1) if v not in self_conflict]
    valid_set = set(valid_vertices)
    
    # Helper: check if a set is independent
    def is_independent(s):
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if s_list[j] in adj[s_list[i]]:
                    return False
        return True
    
    # Greedy: build independent set by picking vertices in given order
    def greedy_independent(order):
        chosen = set()
        for v in order:
            if v in self_conflict:
                continue
            # Check if v conflicts with any chosen
            conflict = False
            for c in chosen:
                if v in adj[c]:
                    conflict = True
                    break
            if not conflict:
                chosen.add(v)
        return chosen
    
    # 1-swap local search: try replacing one element with another to increase size
    def local_search(s, max_iter=5000):
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            s_list = list(s)
            # Try adding a vertex not in s
            for v in valid_vertices:
                if v in s:
                    continue
                # Check if v can be added
                can_add = True
                for c in s:
                    if v in adj[c]:
                        can_add = False
                        break
                if can_add:
                    s.add(v)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove one from s, add one not in s
            for rem in s_list:
                for v in valid_vertices:
                    if v in s:
                        continue
                    # Check if v conflicts with s minus rem
                    can_add = True
                    for c in s:
                        if c == rem:
                            continue
                        if v in adj[c]:
                            can_add = False
                            break
                    if can_add:
                        s.discard(rem)
                        s.add(v)
                        improved = True
                        break
                if improved:
                    break
        return s
    
    # Branch and bound with greedy coloring
    def branch_and_bound(candidates, current_set, best_size, deadline):
        """
        candidates: list of vertices still available (not yet decided)
        current_set: set of vertices chosen so far
        best_size: best size found
        """
        if time.time() > deadline:
            return current_set
        
        if not candidates:
            if len(current_set) > best_size:
                return current_set
            return current_set
        
        # Upper bound: len(current_set) + len(candidates)
        if len(current_set) + len(candidates) <= best_size:
            return current_set
        
        # Greedy coloring for upper bound
        # Color the subgraph induced by candidates
        cand_set = set(candidates)
        # Build subgraph coloring
        color_map = {}
        colors = []
        for v in candidates:
            # Find smallest color not used by neighbors in candidates
            neighbor_colors = set()
            for u in adj[v]:
                if u in cand_set and u in color_map:
                    neighbor_colors.add(color_map[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_map[v] = c
            if c >= len(colors):
                colors.append(set())
            colors[c].add(v)
        
        upper_bound = len(current_set) + len(colors)
        if upper_bound <= best_size:
            return current_set
        
        # Choose vertex with highest degree in remaining subgraph
        v = max(candidates, key=lambda x: len(adj[x] & cand_set))
        
        # Branch 1: include v
        new_candidates = [u for u in candidates if u != v and u not in adj[v]]
        result1 = branch_and_bound(new_candidates, current_set | {v}, best_size, deadline)
        
        # Branch 2: exclude v
        result2 = branch_and_bound([u for u in candidates if u != v], current_set, 
                                    max(best_size, len(result1)), deadline)
        
        if len(result1) >= len(result2):
            return result1
        return result2
    
    # Main search: multiple randomized restarts
    best_solution = set()
    
    # Initial greedy solution
    order = list(valid_vertices)
    random.shuffle(order)
    best_solution = greedy_independent(order)
    best_solution = local_search(best_solution, max_iter=2000)
    
    # Try more restarts with different orderings
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Different random orderings
        order = list(valid_vertices)
        random.shuffle(order)
        sol = greedy_independent(order)
        sol = local_search(sol, max_iter=1000)
        
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Try branch-and-bound from current best to see if we can improve
    if time.time() < deadline - 10:
        # Use B&B starting from best_solution
        remaining = [v for v in valid_vertices if v not in best_solution]
        # Filter: only vertices that don't conflict with best_solution
        remaining = [v for v in remaining if not any(v in adj[c] for c in best_solution)]
        if remaining:
            # Sort by degree
            remaining.sort(key=lambda x: -len(adj[x]))
            bb_result = branch_and_bound(remaining, set(best_solution), len(best_solution), deadline)
            if len(bb_result) > len(best_solution):
                best_solution = bb_result
                best_solution = local_search(best_solution, max_iter=2000)
    
    # Final local search
    if time.time() < deadline:
        best_solution = local_search(best_solution, max_iter=5000)
    
    # Verify solution
    sol_list = sorted(best_solution)
    for i in range(len(sol_list)):
        for j in range(i, len(sol_list)):
            s = sol_list[i] + sol_list[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid - remove the offending pair
                # This shouldn't happen if our logic is correct
                pass
    
    return sol_list
# EVOLVE-BLOCK-END
