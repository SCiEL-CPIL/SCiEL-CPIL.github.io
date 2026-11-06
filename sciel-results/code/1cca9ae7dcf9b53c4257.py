import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    vertices = list(range(n))
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for faster lookup if needed, but we will just iterate.
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Greedy Construction to get a good initial solution
    # This is the same as the current program's logic, which is fast and effective.
    def greedy_construct():
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                continue
            
            valid = True
            # Check pairs (u, w) and (v, w)
            # w is a neighbor of v
            mask_v = adj[v]
            mask_u = adj[u]
            
            # For each w in N(v), check if common(u, w) + 1 > 2
            # i.e., common(u, w) >= 2
            # common(u, w) = (adj[u] & adj[w]).bit_count()
            
            # Iterate w in N(v)
            # We can iterate bits in mask_v
            # But n=20 is small, just loop
            for w in range(n):
                if w == u or w == v:
                    continue
                if (mask_v >> w) & 1:
                    if (adj[u] & adj[w]).bit_count() >= 2:
                        valid = False
                        break
                if valid and (mask_u >> w) & 1:
                    if (adj[v] & adj[w]).bit_count() >= 2:
                        valid = False
                        break
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    res.append([i, j])
        return res, adj, count

    # Run a few greedy restarts to set the initial best
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        edges, adj, count = greedy_construct()
        if count > best_count:
            best_count = count
            best_edges = edges

    # Phase 2: Branch and Bound to improve
    # We use the P3 budget constraint: sum(C(d_i, 2)) <= 380
    # and the pairwise common neighbor constraint.
    
    # We will try to add edges in a specific order.
    # To make B&B effective, we should order edges such that "conflicting" edges are close together?
    # Or just use a fixed order and rely on pruning.
    
    # Let's use a recursive search.
    # State: 
    #   idx: index in the sorted list of edges to consider
    #   adj: list of bitmasks
    #   degrees: list of degrees
    #   p3_sum: sum of C(d_i, 2)
    #   current_count: number of edges added
    
    # To speed up, we can pre-sort edges.
    # A good ordering might be by vertex index.
    
    # We'll use the full list of edges.
    # We need to check if adding edge (u, v) is valid.
    
    def is_valid_add(adj, u, v):
        # Check if (u, v) can be added
        # 1. Common neighbors of u and v must be <= 2
        #    (If we add it, they become neighbors, but the constraint is on common neighbors of *other* pairs?
        #    No, the constraint is: no two vertices have 3 common neighbors.
        #    If we add (u, v), does it create a pair with 3 common neighbors?
        #    It increases common neighbors for pairs (u, w) and (v, w) where w is neighbor of the other.
        #    And it doesn't change common neighbors for other pairs.
        #    Wait, (u, v) themselves: their common neighbors are the intersection of their *other* neighbors.
        #    This count doesn't change when we add the edge between them.
        #    So we only need to check pairs involving u or v.
        
        # Check pairs (u, w) for w in N(v)
        # New common(u, w) = old common(u, w) + 1 (because v is now in N(u) and v is in N(w))
        # We need new common <= 2 => old common <= 1
        mask_v = adj[v]
        mask_u = adj[u]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if (mask_v >> w) & 1:
                # w is in N(v)
                # v will be in N(u)
                # So v is a common neighbor of u and w
                # Old common neighbors of u and w: adj[u] & adj[w]
                # If (adj[u] & adj[w]).bit_count() >= 2, then adding v makes it >= 3. Invalid.
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            
            if (mask_u >> w) & 1:
                # w is in N(u)
                # u will be in N(v)
                # So u is a common neighbor of v and w
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
        
        return True

    # Upper bound function
    # We have p3_sum. Max allowed is 380.
    # Remaining edges: count of edges in all_edges[idx:]
    # Simple bound: current_count + remaining_edges
    # Tighter bound: 
    #   Each edge added increases p3_sum by d_u + d_v.
    #   To minimize the increase, we would add edges between low-degree vertices.
    #   But we don't know which edges we will add.
    #   We can assume we add edges that increase p3_sum by the minimum possible amount.
    #   The minimum increase for any edge is 0 (if d_u=0, d_v=0)? No, min degree is 0, so increase is 0.
    #   But degrees grow.
    #   This bound is hard to compute quickly.
    #   Let's use a heuristic: 
    #   If p3_sum > 380, prune.
    #   If current_count + remaining_edges <= best_count, prune.
    
    # To improve the bound, we can note that if the graph is dense, degrees are high.
    # Let's just use the simple bound and rely on the validity check to prune deep.
    # With n=20, the search tree might still be large.
    # We can add a time check in the recursion.
    
    # To make it faster, we can try to add edges in an order that leads to high counts early?
    # Or just use the order (0,1), (0,2), ..., (0,19), (1,2)...
    
    # Let's use a randomized order for the B&B as well, with multiple restarts?
    # No, B&B is deterministic for a given order.
    # We can try a few different orders.
    
    def branch_and_bound(edge_order, initial_adj, initial_degrees, initial_p3_sum, initial_count, initial_idx):
        nonlocal best_count, best_edges, deadline
        
        # If we don't have an initial state, start with empty
        if initial_adj is None:
            adj = [0] * n
            degrees = [0] * n
            p3_sum = 0
            count = 0
            idx = 0
            edge_order = list(all_edges)
            random.shuffle(edge_order) # Random order for diversity
        else:
            adj = initial_adj
            degrees = initial_degrees
            p3_sum = initial_p3_sum
            count = initial_count
            idx = initial_idx
            edge_order = edge_order
            
        # Recursion
        def dfs(idx, adj, degrees, p3_sum, count):
            nonlocal best_count, best_edges, deadline
            
            if time.time() > deadline - 5:
                return
            
            if idx == len(edge_order):
                if count > best_count:
                    best_count = count
                    # Reconstruct edges
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_edges = edges
                return
            
            # Pruning
            # 1. Simple upper bound
            if count + (len(edge_order) - idx) <= best_count:
                return
            
            # 2. P3 budget check (if we exceeded, we wouldn't be here, but check for safety)
            # We can also estimate if we can even fit one more edge?
            # If p3_sum is close to 380, we can't add many edges.
            # Min increase in p3_sum for any edge is min(degrees[u] + degrees[v]).
            # If p3_sum + min_increase > 380, then we can't add any more edges?
            # Not exactly, we can add edges that don't increase p3_sum? No, every edge increases it by d_u + d_v >= 0.
            # If all remaining edges have d_u + d_v > 380 - p3_sum, we can't add them.
            
            u, v = edge_order[idx]
            
            # Option 1: Skip this edge
            dfs(idx + 1, adj, degrees, p3_sum, count)
            
            # Option 2: Add this edge
            if is_valid_add(adj, u, v):
                # Check P3 budget
                new_p3_sum = p3_sum + degrees[u] + degrees[v]
                if new_p3_sum <= 380:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    
                    dfs(idx + 1, adj, degrees, new_p3_sum, count + 1)
                    
                    # Backtrack
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    degrees[u] -= 1
                    degrees[v] -= 1

        dfs(0, adj, degrees, p3_sum, count)

    # Run B&B with a few different random orderings
    # The greedy phase already found a good solution. B&B can try to beat it.
    
    # To speed up B&B, we can start from the greedy solution and try to swap/add edges?
    # That's local search. B&B from scratch is exhaustive but pruned.
    # Given n=20, B&B from scratch might be too slow if not pruned well.
    # However, the P3 budget is a very strong constraint.
    # Let's try running B&B for a limited time.
    
    # We'll run B&B in a loop with different random seeds for edge ordering.
    # But B&B is expensive.
    # Maybe just one run with a smart ordering?
    
    # Let's try to sort edges by some heuristic.
    # Edges between high-degree vertices are more "expensive" in P3 budget.
    # We want to find a dense graph, so we want high degrees.
    # This is tricky.
    
    # Let's just run the greedy more aggressively and use a simple local search (swap) to improve.
    # The current program's score is 1.96. The inspiration (2.06) likely used a better search.
    # The MCIS idea was rejected because it's not a simple IS problem.
    # But we can use the B&B with P3 pruning.
    
    # Let's refine the B&B to be faster.
    # We can precompute the list of edges for the current state? No, it's dynamic.
    
    # Let's try 5 B&B runs with different random edge orderings.
    # Each run has a time limit of (deadline - current_time) / 5.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_bb_runs = min(5, int(remaining_time / 20) + 1) # At least 1, max 5
        for _ in range(num_bb_runs):
            if time.time() > deadline - 5:
                break
            # Random order
            order = list(all_edges)
            random.shuffle(order)
            # Run B&B
            branch_and_bound(order, None, None, None, None, None)

    if not best_edges:
        # Fallback
        for i in range(n):
            best_edges.append([i, (i + 1) % n])

    return best_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")
