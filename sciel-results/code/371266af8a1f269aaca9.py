import time
import random
from math import gcd
from functools import lru_cache
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    num_set = set(numbers)
    
    # 1. Identify all bad triples
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = a * b // gcd(a, b)
            for c in range(b + 1, n + 1):
                lac = a * c // gcd(a, c)
                if lac != lab:
                    continue
                lbc = b * c // gcd(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers
    
    # 2. Precompute conflict structures
    # For each vertex x, conflicts[x] is a list of pairs (u,v) such that (x,u,v) is a bad triple
    conflicts = {x: [] for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Build 2-shadow conflict graph: edge (u,v) if there exists w such that (u,v,w) is bad
    conflict_graph = {x: set() for x in numbers}
    for a, b, c in bad_triples:
        conflict_graph[a].add(b)
        conflict_graph[a].add(c)
        conflict_graph[b].add(a)
        conflict_graph[b].add(c)
        conflict_graph[c].add(a)
        conflict_graph[c].add(b)
    
    # Helper function to check if x can be added to current_set
    def can_add(x, current_set):
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True
    
    # Compute degree in 2-shadow for each vertex
    shadow_degree = {x: len(conflict_graph[x]) for x in numbers}
    
    # Compute hypergraph degree (number of bad triples containing x)
    hyper_degree = {x: len(conflicts[x]) for x in numbers}
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Multi-start Greedy + Local Search
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
            
        S = set(numbers)
        
        # Greedy removal using hypergraph degree
        while True:
            if time.time() > deadline - 25.0:
                break
                
            max_deg = 0
            candidates = []
            for x in S:
                # Count active conflicts: pairs (u,v) in conflicts[x] where both u,v in S
                active_conflicts = 0
                for u, v in conflicts[x]:
                    if u in S and v in S:
                        active_conflicts += 1
                
                if active_conflicts > max_deg:
                    max_deg = active_conflicts
                    candidates = [x]
                elif active_conflicts == max_deg and max_deg > 0:
                    candidates.append(x)
            
            if not candidates or max_deg == 0:
                break
                
            target = random.choice(candidates)
            S.remove(target)
        
        # Local search: try to add back vertices
        improved = True
        local_iter = 0
        while improved and local_iter < 100:
            improved = False
            local_iter += 1
            if time.time() > deadline - 25.0:
                break
            
            addable = []
            for x in numbers:
                if x not in S and can_add(x, S):
                    addable.append(x)
            
            if not addable:
                break
            
            # Choose the addable vertex that creates the fewest new conflicts
            best_add = None
            min_new_conflicts = float('inf')
            for x in addable:
                # Count how many pairs in conflicts[x] have both elements in S
                new_conflicts = 0
                for u, v in conflicts[x]:
                    if u in S and v in S:
                        new_conflicts += 1
                if new_conflicts < min_new_conflicts:
                    min_new_conflicts = new_conflicts
                    best_add = x
            
            if best_add is not None:
                S.add(best_add)
                improved = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 2: Branch and Bound with tight upper bounds
    # Use 2-shadow matching for upper bound
    
    def compute_upper_bound(available, current_size):
        """Compute upper bound on independent set size using matching on 2-shadow."""
        if not available:
            return current_size
        
        # Greedy matching on 2-shadow induced subgraph
        matched = set()
        matching_size = 0
        
        # Get edges in induced subgraph
        edges = []
        for u in available:
            for v in conflict_graph[u]:
                if v in available and u < v:
                    edges.append((u, v))
        
        # Sort by degree sum for better matching
        edges.sort(key=lambda e: len(conflict_graph[e[0]] & available) + len(conflict_graph[e[1]] & available), reverse=True)
        
        for u, v in edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                matching_size += 1
        
        # Upper bound: current_size + |available| - matching_size
        # (must remove at least matching_size vertices to break all matched edges)
        return current_size + len(available) - matching_size
    
    # Branch and bound
    def branch_and_bound(available, current_set, current_size, depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline - 3.0:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Pruning: simple bound
        if current_size + len(available) <= best_size:
            return
        
        # Compute upper bound using matching
        upper_bound = compute_upper_bound(available, current_size)
        if upper_bound <= best_size:
            return
        
        # Choose pivot: vertex with highest shadow degree in available
        max_deg = -1
        pivot = None
        for x in available:
            deg = len(conflict_graph[x] & available)
            if deg > max_deg:
                max_deg = deg
                pivot = x
        
        if pivot is None or max_deg == 0:
            # No conflicts, take all
            if current_size + len(available) > best_size:
                best_size = current_size + len(available)
                best_set = set(current_set) | set(available)
            return
        
        # Branch 1: Include pivot
        if can_add(pivot, current_set):
            new_available = set(available)
            new_available.remove(pivot)
            # Remove vertices that would create bad triples with pivot and existing set
            # For each pair (u,v) in conflicts[pivot], if u in current_set and v in new_available, remove v
            # and if v in current_set and u in new_available, remove u
            to_remove = set()
            for u, v in conflicts[pivot]:
                if u in current_set and v in new_available:
                    to_remove.add(v)
                elif v in current_set and u in new_available:
                    to_remove.add(u)
            new_available -= to_remove
            branch_and_bound(new_available, current_set | {pivot}, current_size + 1, depth + 1)
        
        # Branch 2: Exclude pivot
        new_available = set(available)
        new_available.remove(pivot)
        branch_and_bound(new_available, current_set, current_size, depth + 1)
    
    # Run branch and bound with time limit
    available = set(numbers)
    current_set = set(best_set) if best_set else set()
    current_size = len(current_set)
    
    # Remove vertices already in current_set from available
    available -= current_set
    
    branch_and_bound(available, current_set, current_size, 0)
    
    if best_size == 0:
        return [1]
        
    return list(best_set)
