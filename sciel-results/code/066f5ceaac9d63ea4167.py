import time
import random

def _has_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    for v0 in range(n):
        nbrs0 = adj[v0]
        if len(nbrs0) < 2:
            continue
        nbrs_list = list(nbrs0)
        for i in range(len(nbrs_list)):
            v1 = nbrs_list[i]
            for j in range(i + 1, len(nbrs_list)):
                v4 = nbrs_list[j]
                found = False
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    common = adj[v2] & adj[v4]
                    if v0 in common:
                        common.discard(v0)
                    if v1 in common:
                        common.discard(v1)
                    if common:
                        found = True
                        break
                if found:
                    return True
    return False

def _alpha_ge_9(adj, n):
    """Check if independence number >= 9 using greedy + backtracking."""
    # Greedy lower bound
    remaining = set(range(n))
    independent = []
    while remaining:
        min_deg = float('inf')
        min_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        independent.append(min_v)
        remaining.discard(min_v)
        remaining -= adj[min_v]
        if len(independent) >= 9:
            return True

    # Branch and bound
    def backtrack(remaining_set, current_independent):
        if len(current_independent) >= 9:
            return True
        if not remaining_set:
            return False
        if len(current_independent) + len(remaining_set) < 9:
            return False
        
        # Pick a vertex with minimum degree in remaining
        v = None
        min_deg = float('inf')
        for x in remaining_set:
            deg = len(adj[x] & remaining_set)
            if deg < min_deg:
                min_deg = deg
                v = x
        
        # Branch 1: include v
        new_remaining = remaining_set - {v} - adj[v]
        if backtrack(new_remaining, current_independent + [v]):
            return True
        
        # Branch 2: exclude v
        new_remaining2 = remaining_set - {v}
        if backtrack(new_remaining2, current_independent):
            return True
        
        return False
    
    return backtrack(set(range(n)), [])

def _find_c5_cycle(adj, n):
    """Returns a list of 5 vertices forming a C5, or None."""
    for v0 in range(n):
        nbrs0 = list(adj[v0])
        if len(nbrs0) < 2:
            continue
        for i in range(len(nbrs0)):
            v1 = nbrs0[i]
            for j in range(i + 1, len(nbrs0)):
                v4 = nbrs0[j]
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v4:
                        continue
                    common = adj[v2] & adj[v4]
                    if v0 in common:
                        common.discard(v0)
                    if v1 in common:
                        common.discard(v1)
                    if common:
                        v3 = next(iter(common))
                        # Verify cycle v0-v1-v2-v3-v4-v0
                        if v0 in adj[v1] and v1 in adj[v2] and v2 in adj[v3] and v3 in adj[v4] and v4 in adj[v0]:
                            # Check distinctness
                            if len({v0, v1, v2, v3, v4}) == 5:
                                return [v0, v1, v2, v3, v4]
    return None

def _find_independent_set_9(adj, n):
    """Returns a list of 9 independent vertices, or None."""
    # Simple greedy might not find it if it exists but is hard to find
    # Use the backtracking logic but return the set
    def backtrack(remaining_set, current_independent):
        if len(current_independent) >= 9:
            return current_independent[:9]
        if not remaining_set:
            return None
        if len(current_independent) + len(remaining_set) < 9:
            return None
        
        v = None
        min_deg = float('inf')
        for x in remaining_set:
            deg = len(adj[x] & remaining_set)
            if deg < min_deg:
                min_deg = deg
                v = x
        
        # Branch 1: include v
        new_remaining = remaining_set - {v} - adj[v]
        res = backtrack(new_remaining, current_independent + [v])
        if res:
            return res
        
        # Branch 2: exclude v
        new_remaining2 = remaining_set - {v}
        res = backtrack(new_remaining2, current_independent)
        if res:
            return res
        
        return None
    
    return backtrack(set(range(n)), [])

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Known 32-vertex solution: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    # Try to find a 33-vertex solution using local search
    random.seed(42)
    
    n_target = 33
    # Initialize adj for 33 vertices
    adj = [set() for _ in range(n_target)]
    
    # Start with 32-vertex graph
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Add vertex 32 with random connections
    # We want to start close to valid, so maybe connect to few vertices?
    # Or many?
    # Let's try connecting to a random subset of size 10
    S_init = set(random.sample(range(32), 10))
    for v in S_init:
        adj[32].add(v)
        adj[v].add(32)
    
    # Local Search
    # Score: number of C5s + 100 * number of independent sets of size 9 (approx)
    # We will just try to fix violations.
    
    for step in range(20000):
        if time.time() > deadline - 5:
            break
        
        # Check C5
        cycle = _find_c5_cycle(adj, n_target)
        if cycle:
            # Remove a random edge from the cycle
            # Edges in cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
            edge_idx = random.randint(0, 4)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % 5]
            adj[u].discard(v)
            adj[v].discard(u)
            continue
            
        # Check Independent Set
        indep_set = _find_independent_set_9(adj, n_target)
        if indep_set:
            # Add a random edge within the independent set
            u = random.choice(indep_set)
            v = random.choice([x for x in indep_set if x != u])
            if u not in adj[v]:
                adj[u].add(v)
                adj[v].add(u)
            continue
            
        # If no C5 and no independent set of size 9, we found a valid graph!
        # Verify exactly
        if not _has_c5(adj, n_target) and not _alpha_ge_9(adj, n_target):
            new_edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        new_edges.append([u, v])
            best_n = 33
            best_edges = new_edges
            break
        
        # Random perturbation to escape local minima
        if random.random() < 0.1:
            # Flip a random edge
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u != v:
                if u in adj[v]:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
    
    return {'n': best_n, 'edges': best_edges}
