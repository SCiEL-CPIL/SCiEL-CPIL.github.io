import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin (evaluator kills at 330s)

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a bitmask of neighbors of i (Python int)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_n = n
    best_adj = adj[:]

    def count_k4(mask, adj, n):
        """Count number of K4s in the subgraph induced by neighbors defined by mask."""
        neighbors = [i for i in range(n) if (mask >> i) & 1]
        count = 0
        if len(neighbors) >= 4:
            for combo in itertools.combinations(neighbors, 4):
                a, b, c, d = combo
                if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[a] >> d) & 1) and \
                   ((adj[b] >> c) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1):
                    count += 1
        return count

    def count_ind4(mask, adj, n):
        """Count number of independent sets of size 4 in the subgraph induced by non-neighbors defined by mask."""
        non_neighbors = [i for i in range(n) if not ((mask >> i) & 1)]
        count = 0
        if len(non_neighbors) >= 4:
            for combo in itertools.combinations(non_neighbors, 4):
                a, b, c, d = combo
                if not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
                   not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and not ((adj[c] >> d) & 1):
                    count += 1
        return count

    def total_violations(mask, adj, n):
        return count_k4(mask, adj, n) + count_ind4(mask, adj, n)

    def local_search_add_vertex(adj, n, max_iters=1000, time_limit=5.0):
        """
        Attempt to find a valid mask for a new vertex using local search.
        Returns (True, mask) if found, (False, None) otherwise.
        """
        start = time.time()
        # Start with a random mask
        current_mask = 0
        for i in range(n):
            if random.random() < 0.5:
                current_mask |= (1 << i)
        
        current_violations = total_violations(current_mask, adj, n)
        if current_violations == 0:
            return True, current_mask

        best_mask = current_mask
        best_violations = current_violations

        for _ in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Pick a random vertex to flip
            u = random.randint(0, n - 1)
            new_mask = current_mask ^ (1 << u)
            new_violations = total_violations(new_mask, adj, n)
            
            # Accept if better or equal (with some probability for equal to allow wandering)
            if new_violations < current_violations or (new_violations == current_violations and random.random() < 0.5):
                current_mask = new_mask
                current_violations = new_violations
                
                if current_violations == 0:
                    return True, current_mask
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_mask = current_mask

        return False, best_mask

    def add_vertex(adj, new_mask):
        n = len(adj)
        new_adj = adj[:]
        new_adj.append(0)
        for i in range(n):
            if (new_mask >> i) & 1:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj

    # Main Search Loop
    best_overall_n = best_n
    best_overall_adj = best_adj[:]
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_adj = best_overall_adj[:]
        current_n = best_overall_n
        
        random.seed(1000 + restart * 13)
        
        # Try to add vertices one by one
        while current_n < 46:
            if time.time() > deadline:
                break
            
            found = False
            # Try multiple local search instances
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                success, mask = local_search_add_vertex(current_adj, current_n, max_iters=500, time_limit=2.0)
                if success:
                    current_adj = add_vertex(current_adj, mask)
                    current_n += 1
                    found = True
                    if current_n > best_overall_n:
                        best_overall_n = current_n
                        best_overall_adj = current_adj[:]
                    break
            
            if not found:
                # If we can't add a vertex, break out of this restart
                # We could implement backtracking here, but for now, we just restart
                break

    # Reconstruct edges from best graph
    best_edges = []
    for i in range(best_overall_n):
        for j in range(i + 1, best_overall_n):
            if (best_overall_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_overall_n, best_edges
