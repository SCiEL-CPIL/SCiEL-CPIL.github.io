import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_optimize(V, time_limit):
        S = set(V)
        removed = list(V)
        
        # 1. Greedy Removal with efficient conflict detection
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break
            
            # Count conflicts efficiently
            conflicts = {}
            S_list = list(S)
            S_set = S
            n = len(S_list)
            for i in range(n):
                a = S_list[i]
                for j in range(i, n):
                    b = S_list[j]
                    s = a + b
                    if s in S_set:
                        conflicts[a] = conflicts.get(a, 0) + 1
                        if b != a:
                            conflicts[b] = conflicts.get(b, 0) + 1
                        conflicts[s] = conflicts.get(s, 0) + 1
            
            if not conflicts:
                break
                
            # Pick element with maximum conflicts
            max_deg = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_deg]
            x_to_remove = random.choice(candidates)
            
            S.remove(x_to_remove)
            removed.remove(x_to_remove)

        # 2. Hill Climbing with improved can_add check
        def can_add(x, S_set, S_list):
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True

        # Multiple passes of hill climbing
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            random.shuffle(removed)
            S_list = list(S)
            S_set = S
            
            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue
                
                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                else:
                    new_removed.append(x)
            
            if improved:
                removed = new_removed
        
        return list(S)

    def build_conflict_graph(V):
        """Build adjacency list for conflict graph where edge means two elements can't both be in sum-free set."""
        V_set = set(V)
        n = len(V)
        adj = [set() for _ in range(n)]
        
        # For each triple (a, b, c) where a+b=c and all in V,
        # a and b conflict, a and c conflict, b and c conflict
        for i in range(n):
            for j in range(i, n):
                s = V[i] + V[j]
                if s in V_set:
                    # Find index of s
                    for k in range(n):
                        if V[k] == s:
                            adj[i].add(j)
                            adj[i].add(k)
                            adj[j].add(i)
                            adj[j].add(k)
                            adj[k].add(i)
                            adj[k].add(j)
                            break
        
        return adj

    def mis_branch_and_bound(V, adj, time_limit):
        """Find maximum independent set using branch and bound."""
        n = len(V)
        if n == 0:
            return set()
        
        # Convert to 0-indexed
        best_size = 0
        best_set = set()
        
        # Use bitsets for efficiency
        # adj_bits[i] is a bitmask of neighbors of i
        adj_bits = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_bits[i] |= (1 << j)
        
        def greedy_coloring(remaining):
            """Greedy coloring to get upper bound on MIS size."""
            # remaining is a set of vertex indices
            if not remaining:
                return 0
            colors = {}
            color_classes = []
            uncolored = set(remaining)
            
            while uncolored:
                current_color = set()
                for v in list(uncolored):
                    # Check if v conflicts with any vertex in current_color
                    if not any((adj_bits[v] >> u) & 1 for u in current_color):
                        current_color.add(v)
                        uncolored.discard(v)
                color_classes.append(current_color)
            
            return len(color_classes)
        
        def branch(remaining, current_set):
            nonlocal best_size, best_set
            
            if time.time() > start_time + time_limit:
                return
            
            if not remaining:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Upper bound: current_size + greedy coloring of remaining
            upper_bound = len(current_set) + greedy_coloring(remaining)
            if upper_bound <= best_size:
                return
            
            # Choose vertex with highest degree in remaining
            max_deg = -1
            pivot = None
            for v in remaining:
                deg = sum(1 for u in remaining if (adj_bits[v] >> u) & 1)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            if pivot is None:
                return
            
            # Branch 1: Include pivot
            new_remaining = set()
            for v in remaining:
                if v != pivot and not ((adj_bits[pivot] >> v) & 1):
                    new_remaining.add(v)
            current_set.add(pivot)
            branch(new_remaining, current_set)
            current_set.discard(pivot)
            
            # Branch 2: Exclude pivot
            remaining.discard(pivot)
            branch(remaining, current_set)
            remaining.add(pivot)
        
        initial_remaining = set(range(n))
        branch(initial_remaining, set())
        
        # Convert indices back to values
        return {V[i] for i in best_set}

    results = []
    
    # Process instances with adaptive restarts based on remaining time
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Try MIS solver first if time permits
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        
        # Allocate time for MIS based on instance size
        if size <= 60:
            mis_time = min(30.0, remaining_time * 0.5)
        elif size <= 120:
            mis_time = min(40.0, remaining_time * 0.4)
        else:
            mis_time = min(50.0, remaining_time * 0.3)
        
        # Try exact MIS solver
        if mis_time > 5.0:
            try:
                adj = build_conflict_graph(V)
                mis_result = mis_branch_and_bound(V, adj, mis_time)
                if len(mis_result) > best_size:
                    best_size = len(mis_result)
                    best_set = list(mis_result)
            except Exception:
                pass
        
        # Fall back to heuristic if MIS didn't find good solution or timed out
        if best_size == 0 or (time.time() - start_time) < DEADLINE - 10:
            elapsed = time.time() - start_time
            remaining = DEADLINE - elapsed
            
            if size <= 60:
                max_restarts = 3
                base_time = 8.0
            elif size <= 120:
                max_restarts = 2
                base_time = 12.0
            else:
                max_restarts = 2
                base_time = 20.0
            
            for r in range(max_restarts):
                if time.time() > start_time + DEADLINE - 5:
                    break
                
                time_per_restart = base_time * (0.7 ** r)
                
                res = greedy_optimize(V, time_per_restart)
                if len(res) > best_size:
                    best_size = len(res)
                    best_set = res
            
            # Verify sum-free
            if not is_sum_free(best_set):
                # Fallback: just use greedy removal
                S = set(V)
                while not is_sum_free(S):
                    conflicts = {}
                    S_list = list(S)
                    S_set = S
                    n = len(S_list)
                    for i in range(n):
                        a = S_list[i]
                        for j in range(i, n):
                            b = S_list[j]
                            s = a + b
                            if s in S_set:
                                conflicts[a] = conflicts.get(a, 0) + 1
                                if b != a:
                                    conflicts[b] = conflicts.get(b, 0) + 1
                                conflicts[s] = conflicts.get(s, 0) + 1
                    if not conflicts:
                        break
                    max_deg = max(conflicts.values())
                    candidates = [x for x, c in conflicts.items() if c == max_deg]
                    x_to_remove = random.choice(candidates)
                    S.remove(x_to_remove)
                best_set = list(S)
        
        results.append(best_set)

    return results
