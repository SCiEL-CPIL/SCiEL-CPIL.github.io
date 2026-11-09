import math
import random
import time

def construct():
    N = 512
    nodes = list(range(1, N + 1))
    
    # Precompute conflict matrix
    conflicts = [set() for _ in range(N + 1)]
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if a * b // math.gcd(a, b) > N:
                conflicts[a].add(b)
                conflicts[b].add(a)

    def greedy_max_independent_set(candidates, forbidden):
        s = set()
        # Sort by degree in the induced subgraph? 
        # For speed, just iterate.
        # To make it deterministic, we can sort by value.
        # But random order might find better sets?
        # Let's stick to input order or sorted by degree.
        # In local_improve, we pass a specific list.
        for node in candidates:
            if node in forbidden:
                continue
            if s.isdisjoint(conflicts[node]) and forbidden.isdisjoint(conflicts[node]):
                s.add(node)
        return s

    def local_improve(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            current_size = len(s)
            
            # Compute releasable sets
            # For each y not in s, find neighbors in s
            releasable = {x: set() for x in s}
            for y in nodes:
                if y in s:
                    continue
                neigh_in_s = conflicts[y].intersection(s)
                if len(neigh_in_s) == 1:
                    x = next(iter(neigh_in_s))
                    releasable[x].add(y)
            
            # Try removing each x
            for x in list(s):
                candidates = releasable[x]
                if not candidates:
                    continue
                
                s_minus_x = s - {x}
                
                # Run greedy on candidates
                # We want to maximize the number of added nodes.
                # Greedy is order-dependent.
                # Let's try a few random orders or just one.
                # Since candidates are disjoint from s_minus_x (by construction of releasable? 
                # Wait. releasable[x] contains y such that conflicts[y] & s == {x}.
                # So y is compatible with s \ {x}.
                # But y might conflict with other y' in candidates.
                # So we need an independent set from candidates.
                
                # Greedy on candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                added = greedy_max_independent_set(cand_list, s_minus_x)
                
                new_s = s_minus_x.union(added)
                if len(new_s) > current_size:
                    s = new_s
                    improved = True
                    break # Restart loop with new s
            
        return s

    best_set = set()
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 80
    
    random.seed(42)
    
    # Initial greedy
    nodes_by_degree = sorted(nodes, key=lambda x: len(conflicts[x]))
    s = greedy_max_independent_set(nodes_by_degree, set())
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
        
    # Random restarts
    while time.time() < deadline:
        order = nodes[:]
        random.shuffle(order)
        s = greedy_max_independent_set(order, set())
        
        s = local_improve(s)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
    return list(best_set)
