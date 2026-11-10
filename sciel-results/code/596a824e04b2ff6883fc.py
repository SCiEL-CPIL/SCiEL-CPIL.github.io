def construct():
    import math
    import time
    from fractions import Fraction
    
    deadline = time.time() + 80  # 80s deadline with 10s safety margin
    
    # Build conflict graph
    # Edge (a,b) if lcm(a,b) <= 256
    nodes = list(range(2, 257))
    n = len(nodes)
    node_index = {v: i for i, v in enumerate(nodes)}
    
    # weight[i] = 1/nodes[i]
    weights = [1.0 / nodes[i] for i in range(n)]
    
    # Build adjacency as bitmasks or sets
    # For 255 nodes, use Python sets for adjacency
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            if math.lcm(a, b) <= 256:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy initial solution
    def greedy_solution():
        selected = set()
        # Sort by weight descending (smaller numbers first)
        order = sorted(range(n), key=lambda i: -weights[i])
        for i in order:
            # Check if i conflicts with any selected
            if not (adj[i] & selected):
                selected.add(i)
        return selected
    
    best_set = greedy_solution()
    best_val = sum(weights[i] for i in best_set)
    
    def val_of(s):
        return sum(weights[i] for i in s)
    
    # Local search: 1-swap and 2-swap
    def local_search(s, max_iters=10000):
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # 1-swap: remove one, add one
            cur_val = val_of(s)
            for rem in list(s):
                for add in range(n):
                    if add in s:
                        continue
                    # Check add doesn't conflict with s - {rem}
                    new_s = s - {rem}
                    if not (adj[add] & new_s):
                        new_val = cur_val - weights[rem] + weights[add]
                        if new_val > cur_val + 1e-12:
                            s = new_s | {add}
                            cur_val = new_val
                            improved = True
            # 2-swap: remove two, add two
            if not improved:
                cur_val = val_of(s)
                s_list = list(s)
                for i_idx in range(len(s_list)):
                    for j_idx in range(i_idx + 1, len(s_list)):
                        rem1, rem2 = s_list[i_idx], s_list[j_idx]
                        new_s = s - {rem1, rem2}
                        for add1 in range(n):
                            if add1 in s:
                                continue
                            if adj[add1] & new_s:
                                continue
                            for add2 in range(n):
                                if add2 in s or add2 == add1:
                                    continue
                                if adj[add2] & new_s:
                                    continue
                                if adj[add1] & {add2}:
                                    continue
                                new_val = cur_val - weights[rem1] - weights[rem2] + weights[add1] + weights[add2]
                                if new_val > cur_val + 1e-12:
                                    s = new_s | {add1, add2}
                                    cur_val = new_val
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return s
    
    best_set = local_search(best_set)
    best_val = val_of(best_set)
    
    # Branch and bound
    # State: (candidate_set, forbidden_set, current_weight, selected_set)
    # Use recursive B&B with pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute for each node: its neighbors
    # For B&B, we process nodes in order of decreasing weight
    
    # Order nodes by weight descending
    order = sorted(range(n), key=lambda i: -weights[i])
    order_idx = [0] * n
    for rank, node in enumerate(order):
        order_idx[node] = rank
    
    # For B&B, we consider nodes in this order
    # State: (pos, selected_mask, forbidden_mask, current_weight)
    # But with 255 nodes, bitmasks are too large. Use sets.
    
    # Alternative: use a simpler B&B with greedy upper bound
    
    def upper_bound(remaining, forbidden):
        # Upper bound: sum of weights of nodes in remaining that are not in forbidden
        # and don't conflict with each other (relax: just sum all)
        total = 0.0
        for i in remaining:
            if i not in forbidden:
                total += weights[i]
        return total
    
    # Better: for each node in remaining, if it doesn't conflict with selected, 
    # we can potentially add it. The upper bound is current_weight + sum of weights 
    # of all non-conflicting remaining nodes.
    
    # Actually, let's use a tighter bound: process remaining nodes, and for each,
    # if it can be added (no conflict with selected), include its weight in the bound.
    # This is still an upper bound (relaxing the mutual conflicts among remaining).
    
    def tight_upper_bound(selected, remaining, forbidden):
        bound = 0.0
        for i in remaining:
            if i in forbidden:
                continue
            if i in selected:
                continue
            if not (adj[i] & selected):
                bound += weights[i]
        return bound
    
    # B&B with time limit
    # Use iterative deepening or just recursive with time check
    
    # Given the size, let's try a different approach: 
    # The key insight is that for large numbers, conflicts are rare.
    # Let's identify "free" nodes that don't conflict with anything in a potential solution.
    
    # Actually, let me try a more practical B&B:
    # Sort nodes by weight desc. At each step, decide include/exclude.
    # Prune if upper bound <= best_val.
    
    best_val_global = [best_val]
    best_set_global = [set(best_set)]
    
    def bb(pos, selected, forbidden, current_w):
        if time.time() > deadline:
            return
        if pos >= n:
            if current_w > best_val_global[0] + 1e-12:
                best_val_global[0] = current_w
                best_set_global[0] = set(selected)
            return
        
        node = order[pos]
        
        # Upper bound: current_w + sum of weights of all remaining nodes not in forbidden
        # that don't conflict with selected
        ub = current_w
        for p in range(pos + 1, n):
            rnode = order[p]
            if rnode in forbidden:
                continue
            if rnode in selected:
                continue
            if not (adj[rnode] & selected):
                ub += weights[rnode]
        
        if ub <= best_val_global[0] + 1e-12:
            return
        
        # Option 1: include node
        if node not in forbidden and not (adj[node] & selected):
            new_selected = selected | {node}
            new_forbidden = forbidden | adj[node]
            bb(pos + 1, new_selected, new_forbidden, current_w + weights[node])
        
        # Option 2: exclude node
        bb(pos + 1, selected, forbidden, current_w)
    
    # This B&B is too slow for 255 nodes. Let's use a smarter approach.
    
    # Key insight: Many nodes are "independent" - they don't conflict with each other.
    # Let's identify the structure:
    # - Nodes > 128: two such nodes a,b conflict only if lcm(a,b) <= 256
    #   This requires them to share a large gcd. E.g., 128|a and 128|b, but 128*2=256, 
    #   so only 128 and 256 conflict among multiples of 128.
    # - Actually let me count conflicts per node.
    
    # Let's try a different strategy: 
    # 1. Identify "heavy" nodes (small values, high weight) that have many conflicts
    # 2. For the rest, they're mostly independent
    
    # Count conflicts
    conflict_count = [len(adj[i]) for i in range(n)]
    
    # Nodes with 0 conflicts can always be included
    free_nodes = [i for i in range(n) if len(adj[i]) == 0]
    
    # All free nodes should be in the optimal solution
    # Remove them and their neighbors from consideration? No, their neighbors 
    # just can't be in the solution if the free node is included.
    # But since free nodes have no conflicts, they don't restrict anything.
    
    # Let's just include all free nodes.
    mandatory = set(free_nodes)
    
    # For the remaining nodes, we need to solve MWIS on the subgraph
    # induced by non-free nodes, but with the constraint that we can't 
    # include a node that conflicts with a mandatory node.
    # But mandatory nodes have no conflicts, so they don't conflict with anyone.
    
    # So the problem reduces to MWIS on the subgraph of non-free nodes.
    
    # How many non-free nodes are there?
    non_free = [i for i in range(n) if len(adj[i]) > 0]
    
    # Let's try B&B on the non-free nodes only
    # Sort by weight desc
    non_free_order = sorted(non_free, key=lambda i: -weights[i])
    
    # B&B on non_free_order
    # State: (pos, selected, current_w)
    # selected is a set of indices from non_free
    
    best_nf_val = [0.0]
    best_nf_set = [set()]
    
    # Initial: greedy on non-free
    sel_nf = set()
    for node in non_free_order:
        if not (adj[node] & sel_nf):
            sel_nf.add(node)
    best_nf_val[0] = val_of(sel_nf)
    best_nf_set[0] = set(sel_nf)
    
    # Local search on non-free
    def local_search_nf(s):
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < 5000 and time.time() < deadline:
            improved = False
            iters += 1
            cur_val = val_of(s)
            # 1-swap
            for rem in list(s):
                for add in non_free:
                    if add in s:
                        continue
                    new_s = s - {rem}
                    if not (adj[add] & new_s):
                        new_val = cur_val - weights[rem] + weights[add]
                        if new_val > cur_val + 1e-12:
                            s = new_s | {add}
                            cur_val = new_val
                            improved = True
            # 2-swap
            if not improved:
                cur_val = val_of(s)
                s_list = list(s)
                for i_idx in range(len(s_list)):
                    for j_idx in range(i_idx + 1, len(s_list)):
                        rem1, rem2 = s_list[i_idx], s_list[j_idx]
                        new_s = s - {rem1, rem2}
                        for add1 in non_free:
                            if add1 in s:
                                continue
                            if adj[add1] & new_s:
                                continue
                            for add2 in non_free:
                                if add2 in s or add2 == add1:
                                    continue
                                if adj[add2] & new_s:
                                    continue
                                if add1 in adj[add2]:
                                    continue
                                new_val = cur_val - weights[rem1] - weights[rem2] + weights[add1] + weights[add2]
                                if new_val > cur_val + 1e-12:
                                    s = new_s | {add1, add2}
                                    cur_val = new_val
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return s
    
    best_nf_set[0] = local_search_nf(best_nf_set[0])
    best_nf_val[0] = val_of(best_nf_set[0])
    
    # B&B on non-free nodes
    # Use a smarter upper bound
    
    def bb_nf(pos, selected, current_w):
        if time.time() > deadline:
            return
        if pos >= len(non_free_order):
            if current_w > best_nf_val[0] + 1e-12:
                best_nf_val[0] = current_w
                best_nf_set[0] = set(selected)
            return
        
        node = non_free_order[pos]
        
        # Upper bound: current_w + sum of weights of remaining non-free nodes 
        # that don't conflict with selected
        ub = current_w
        for p in range(pos + 1, len(non_free_order)):
            rnode = non_free_order[p]
            if rnode not in selected and not (adj[rnode] & selected):
                ub += weights[rnode]
        
        if ub <= best_nf_val[0] + 1e-12:
            return
        
        # Include node
        if not (adj[node] & selected):
            bb_nf(pos + 1, selected | {node}, current_w + weights[node])
        
        # Exclude node
        bb_nf(pos + 1, selected, current_w)
    
    # Only run B&B if non-free set is small enough
    if len(non_free) <= 50:
        bb_nf(0, set(), 0.0)
    
    # Combine: mandatory + best non-free
    final_set = mandatory | best_nf_set[0]
    final_val = val_of(final_set)
    
    if final_val > best_val + 1e-12:
        best_set = final_set
        best_val = final_val
    
    # Convert back to values
    result = [nodes[i] for i in best_set]
    result.sort()
    
    # Validate
    assert len(result) > 0
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.lcm(result[i], result[j]) > 256
    
    return result
