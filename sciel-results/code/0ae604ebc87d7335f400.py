import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def incremental_score(values, edge_list, changed_edge=None, old_vals=None):
        """Fast score computation; if changed_edge given, recompute only affected sums/prods."""
        sums = {}
        prods = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums[s] = sums.get(s, 0) + 1
            prods[p] = prods.get(p, 0) + 1
        return max(len(sums), len(prods))

    # Precompute for each vertex the list of edges incident to it
    vertex_edges = [[] for _ in range(n)]
    for e in all_edges:
        vertex_edges[e[0]].append(e)
        vertex_edges[e[1]].append(e)

    def optimize_values_for_edges(vals, edges, budget_iters=50000):
        """
        Fine-tune values for fixed edges with collision-aware moves.
        """
        vals = list(vals)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        
        edge_arr = [(e[0], e[1]) for e in edges]
        
        # Build adjacency for fast neighbor lookup
        adj = [set() for _ in range(n)]
        for u, v in edge_arr:
            adj[u].add(v)
            adj[v].add(u)
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            new_val = None
            
            r = random.random()
            if r < 0.30:
                # Small perturbation
                new_val = old_val + random.randint(-5, 5)
            elif r < 0.50:
                # Medium perturbation
                new_val = old_val + random.randint(-20, 20)
            elif r < 0.65:
                # Random restart in local range
                new_val = random.randint(1, min(10000, max(2, old_val * 3)))
            elif r < 0.80:
                # Collision-driven: pick a random edge (a,b), try to make vals[idx]
                # create a sum collision with that edge for some neighbor
                a, b = random.choice(edge_arr)
                target_sum = vals[a] + vals[b]
                for nb in adj[idx]:
                    cand = target_sum - vals[nb]
                    if 1 <= cand <= 10000 and cand != old_val:
                        if not any(v == cand for j, v in enumerate(vals) if j != idx):
                            new_val = cand
                            break
            else:
                # Product collision: pick random edge, try to match product
                a, b = random.choice(edge_arr)
                target_prod = vals[a] * vals[b]
                for nb in adj[idx]:
                    if vals[nb] != 0 and target_prod % vals[nb] == 0:
                        cand = target_prod // vals[nb]
                        if 1 <= cand <= 10000 and cand != old_val:
                            if not any(v == cand for j, v in enumerate(vals) if j != idx):
                                new_val = cand
                                break
            
            if new_val is None or new_val <= 0 or new_val > 10000:
                continue
            if any(v == new_val for j, v in enumerate(vals) if j != idx):
                continue
            
            new_vals = list(vals)
            new_vals[idx] = new_val
            new_score = score_of(new_vals, edges)
            
            if new_score < cur_score:
                vals = new_vals
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(vals)
            elif new_score == cur_score and random.random() < 0.35:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 2500 and random.random() < 0.015:
                no_improve = 0

        return best_score, best_vals

    def optimize_edges_for_values(vals, edges, budget_iters=50000):
        """
        Optimize edges for fixed values with collision-aware edge selection.
        """
        edges = list(edges)
        edges_set = set(edges)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_edges = list(edges)
        
        # Precompute sum and product for all possible edges
        edge_sum = {}
        edge_prod = {}
        for e in all_edges:
            u, v = e
            edge_sum[e] = vals[u] + vals[v]
            edge_prod[e] = vals[u] * vals[v]
        
        # For fast scoring, track current sum/prod counts
        sum_counts = {}
        prod_counts = {}
        for u, v in edges:
            s = vals[u] + vals[v]
            p = vals[u] * vals[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        def quick_score_from_counts():
            return max(len(sum_counts), len(prod_counts))
        
        no_improve = 0
        it = 0
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            move_type = random.random()
            
            if move_type < 0.75:
                # Single edge swap
                i = random.randint(0, m - 1)
                e_out = edges[i]
                
                # Prefer edges whose sum/prod already appear multiple times (safe to remove)
                # and edges that would create new collisions (good to add)
                candidates = []
                for e_in in all_edges:
                    if e_in in edges_set:
                        continue
                    # Score: does adding e_in create a collision?
                    s_in = edge_sum[e_in]
                    p_in = edge_prod[e_in]
                    collision_score = (1 if sum_counts.get(s_in, 0) > 0 else 0) + \
                                      (1 if prod_counts.get(p_in, 0) > 0 else 0)
                    candidates.append((collision_score, e_in))
                
                if not candidates:
                    continue
                
                # Weighted random selection favoring collisions
                max_cs = max(c[0] for c in candidates)
                # Pick from top candidates
                top = [c for c in candidates if c[0] >= max_cs - 1]
                _, e_in = random.choice(top)
                
                # Apply
                u_out, v_out = e_out
                s_out = vals[u_out] + vals[v_out]
                p_out = vals[u_out] * vals[v_out]
                u_in, v_in = e_in
                s_in = edge_sum[e_in]
                p_in = edge_prod[e_in]
                
                # Update counts
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
                prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
                
                new_score = quick_score_from_counts()
                
                if new_score < cur_score:
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.25:
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    no_improve = 0
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        del sum_counts[s_in]
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        del prod_counts[p_in]
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    no_improve += 1
            else:
                # Two-edge swap
                i1 = random.randint(0, m - 1)
                i2 = random.randint(0, m - 1)
                if i1 == i2:
                    continue
                e_out1 = edges[i1]
                e_out2 = edges[i2]
                
                # Find two good incoming edges
                e_in1 = None
                e_in2 = None
                for _ in range(10):
                    e_in1 = random.choice(all_edges)
                    if e_in1 not in edges_set and e_in1 != e_out1 and e_in1 != e_out2:
                        break
                for _ in range(10):
                    e_in2 = random.choice(all_edges)
                    if e_in2 not in edges_set and e_in2 != e_out1 and e_in2 != e_out2 and e_in2 != e_in1:
                        break
                
                if e_in1 is None or e_in2 is None:
                    continue
                
                u1o, v1o = e_out1
                u2o, v2o = e_out2
                s1o = vals[u1o] + vals[v1o]
                p1o = vals[u1o] * vals[v1o]
                s2o = vals[u2o] + vals[v2o]
                p2o = vals[u2o] * vals[v2o]
                
                u1i, v1i = e_in1
                u2i, v2i = e_in2
                s1i = edge_sum[e_in1]
                p1i = edge_prod[e_in1]
                s2i = edge_sum[e_in2]
                p2i = edge_prod[e_in2]
                
                sum_counts[s1o] -= 1
                if sum_counts[s1o] == 0:
                    del sum_counts[s1o]
                prod_counts[p1o] -= 1
                if prod_counts[p1o] == 0:
                    del prod_counts[p1o]
                sum_counts[s2o] -= 1
                if sum_counts[s2o] == 0:
                    del sum_counts[s2o]
                prod_counts[p2o] -= 1
                if prod_counts[p2o] == 0:
                    del prod_counts[p2o]
                
                sum_counts[s1i] = sum_counts.get(s1i, 0) + 1
                prod_counts[p1i] = prod_counts.get(p1i, 0) + 1
                sum_counts[s2i] = sum_counts.get(s2i, 0) + 1
                prod_counts[p2i] = prod_counts.get(p2i, 0) + 1
                
                new_score = quick_score_from_counts()
                
                if new_score < cur_score:
                    edges[i1] = e_in1
                    edges[i2] = e_in2
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.15:
                    edges[i1] = e_in1
                    edges[i2] = e_in2
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    no_improve = 0
                else:
                    # Revert
                    sum_counts[s1i] -= 1
                    if sum_counts[s1i] == 0:
                        del sum_counts[s1i]
                    prod_counts[p1i] -= 1
                    if prod_counts[p1i] == 0:
                        del prod_counts[p1i]
                    sum_counts[s2i] -= 1
                    if sum_counts[s2i] == 0:
                        del sum_counts[s2i]
                    prod_counts[p2i] -= 1
                    if prod_counts[p2i] == 0:
                        del prod_counts[p2i]
                    
                    sum_counts[s1o] = sum_counts.get(s1o, 0) + 1
                    prod_counts[p1o] = prod_counts.get(p1o, 0) + 1
                    sum_counts[s2o] = sum_counts.get(s2o, 0) + 1
                    prod_counts[p2o] = prod_counts.get(p2o, 0) + 1
                    no_improve += 1
            
            if no_improve > 1500 and random.random() < 0.02:
                no_improve = 0

        return best_score, best_edges

    def joint_search(vals, edges, budget_iters=80000):
        """
        Joint local search on values and edges with collision-aware moves.
        """
        vals = list(vals)
        edges = list(edges)
        edges_set = set(edges)
        
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        no_improve = 0
        it = 0
        
        while it < budget_iters and time.time() < deadline - 3.0:
            it += 1
            
            if random.random() < 0.50:
                # Edge move with collision preference
                i = random.randint(0, m - 1)
                e_out = edges[i]
                
                # Find an edge that creates a sum or product collision
                e_in = None
                for _ in range(20):
                    cand = random.choice(all_edges)
                    if cand in edges_set:
                        continue
                    s_c = vals[cand[0]] + vals[cand[1]]
                    p_c = vals[cand[0]] * vals[cand[1]]
                    # Check if this sum or product already exists among current edges
                    collision = False
                    for u, v in edges:
                        if (u, v) == cand:
                            continue
                        if vals[u] + vals[v] == s_c or vals[u] * vals[v] == p_c:
                            collision = True
                            break
                    if collision:
                        e_in = cand
                        break
                
                if e_in is None:
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue
                
                new_edges = list(edges)
                new_edges[i] = e_in
                new_score = score_of(vals, new_edges)
                
                if new_score < cur_score:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    adj[e_out[0]].discard(e_out[1])
                    adj[e_out[1]].discard(e_out[0])
                    adj[e_in[0]].add(e_in[1])
                    adj[e_in[1]].add(e_in[0])
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.30:
                    edges = new_edges
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    adj[e_out[0]].discard(e_out[1])
                    adj[e_out[1]].discard(e_out[0])
                    adj[e_in[0]].add(e_in[1])
                    adj[e_in[1]].add(e_in[0])
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Value move with collision targeting
                idx = random.randint(0, n - 1)
                old_val = vals[idx]
                new_val = None
                r = random.random()
                
                if r < 0.30:
                    new_val = old_val + random.randint(-5, 5)
                elif r < 0.50:
                    new_val = old_val + random.randint(-15, 15)
                elif r < 0.65:
                    new_val = random.randint(1, min(10000, max(2, old_val * 3)))
                elif r < 0.82:
                    # Sum collision: pick random edge, try to match its sum
                    e_target = random.choice(edges)
                    target_sum = vals[e_target[0]] + vals[e_target[1]]
                    for nb in adj[idx]:
                        cand = target_sum - vals[nb]
                        if 1 <= cand <= 10000 and cand != old_val:
                            if not any(v == cand for j, v in enumerate(vals) if j != idx):
                                new_val = cand
                                break
                else:
                    # Product collision
                    e_target = random.choice(edges)
                    target_prod = vals[e_target[0]] * vals[e_target[1]]
                    for nb in adj[idx]:
                        if vals[nb] != 0 and target_prod % vals[nb] == 0:
                            cand = target_prod // vals[nb]
                            if 1 <= cand <= 10000 and cand != old_val:
                                if not any(v == cand for j, v in enumerate(vals) if j != idx):
                                    new_val = cand
                                    break
                
                if new_val is None or new_val <= 0 or new_val > 10000:
                    continue
                if any(v == new_val for j, v in enumerate(vals) if j != idx):
                    continue
                
                new_vals = list(vals)
                new_vals[idx] = new_val
                new_score = score_of(new_vals, edges)
                
                if new_score < cur_score:
                    vals = new_vals
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                elif new_score == cur_score and random.random() < 0.30:
                    vals = new_vals
                    no_improve = 0
                else:
                    no_improve += 1
            
            if no_improve > 3000 and random.random() < 0.012:
                no_improve = 0

        return best_score, best_vals, best_edges

    def construct_partition_solution(partition, vals):
        """
        Given a partition of vertices into groups, construct edges such that
        intra-group edges have the same sum (if possible) and we fill with
        inter-group edges.
        
        For a group G, if we set values so that all pairs in G have the same sum,
        that requires the group to have at most 2 distinct values (a, b) with a+b = S.
        Actually for 3+ vertices all same sum is impossible unless all values equal.
        So use: groups of 2 with same sum, or groups where we accept multiple sums.
        
        Better: for a group of size k, the number of distinct sums among its edges
        is at most k-1 (if values are in arithmetic progression, sums range from
        v1+v2 to v_{k-1}+v_k, which is k-1 distinct values).
        
        So a group of size 4 gives at most 3 distinct sums for 6 edges.
        """
        # For each group, generate all intra-group edges
        edges = []
        edge_set = set()
        
        for group in partition:
            for u, v in combinations(group, 2):
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    edges.append(e)
                    edge_set.add(e)
        
        # Fill remaining with inter-group edges, preferring those that create collisions
        remaining = m - len(edges)
        if remaining > 0:
            # Sort inter-group edges by sum to encourage collisions
            inter_edges = []
            for e in all_edges:
                if e not in edge_set:
                    inter_edges.append(e)
            
            # Sort by sum then product
            inter_edges.sort(key=lambda e: (vals[e[0]] + vals[e[1]], vals[e[0]] * vals[e[1]]))
            
            for e in inter_edges:
                if remaining <= 0:
                    break
                edges.append(e)
                edge_set.add(e)
                remaining -= 1
        
        return edges[:m]

    def make_arith_group(group, start, diff):
        """Set values for a group as arithmetic progression."""
        vals = {}
        for i, v in enumerate(group):
            vals[v] = start + i * diff
        return vals

    # Strategy: Try various partition structures and value assignments
    # Key: minimize distinct sums AND distinct products
    
    best_score = float('inf')
    best_vals = None
    best_edges = None
    
    random.seed(42)
    
    # Candidate partitions and value strategies
    candidates = []
    
    # Strategy A: Groups of 4,4,3,3 with arithmetic values
    part1 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    vals1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidates.append((part1, vals1))
    
    # Strategy B: Groups of 5,4,3,2
    part2 = [[0,1,2,3,4], [5,6,7,8], [9,10,11], [12,13]]
    vals2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    candidates.append((part2, vals2))
    
    # Strategy C: Divisor-rich values for product collisions
    vals3 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    part3 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part3, vals3))
    
    # Strategy D: Small values for dense sum collisions
    vals4 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    part4 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part4, vals4))
    
    # Strategy E: Mixed - some small, some larger
    vals5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24]
    part5 = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
    candidates.append((part5, vals5))
    
    # Strategy F: Random small values (multiple seeds)
    for _ in range(4):
        rv = random.sample(range(1, 30), n)
        part = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
        candidates.append((part, rv))
    
    # Strategy G: Random medium values
    for _ in range(3):
        rv = random.sample(range(1, 80), n)
        part = [[0,1,2,3], [4,5,6,7], [8,9,10], [11,12,13]]
        candidates.append((part, rv))
    
    # For each candidate, construct edges and optimize
    for idx_cand, (part, vals) in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
        
        vals = list(vals)
        edges = construct_partition_solution(part, vals)
        
        if len(edges) != m:
            continue
        
        # Optimize edges for these values (more iterations for earlier candidates)
        iters_edge = max(5000, 12000 - idx_cand * 500)
        sc, edges_opt = optimize_edges_for_values(vals, edges, budget_iters=iters_edge)
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = edges_opt
        
        # Optimize values for these edges
        if time.time() > deadline - 12.0:
            break
        iters_val = max(4000, 10000 - idx_cand * 400)
        sc, vals_opt = optimize_values_for_edges(vals, edges_opt, budget_iters=iters_val)
        if sc < best_score:
            best_score = sc
            best_vals = vals_opt
            best_edges = edges_opt
        
        # Joint search
        if time.time() > deadline - 8.0:
            break
        iters_joint = max(5000, 15000 - idx_cand * 800)
        sc, vals_j, edges_j = joint_search(vals_opt, edges_opt, budget_iters=iters_joint)
        if sc < best_score:
            best_score = sc
            best_vals = vals_j
            best_edges = edges_j
    
    # Additional: pure random restarts with joint search
    for _ in range(6):
        if time.time() > deadline - 6.0:
            break
        rv = random.sample(range(1, 60), n)
        re = random.sample(all_edges, m)
        sc, vals_j, edges_j = joint_search(rv, re, budget_iters=8000)
        if sc < best_score:
            best_score = sc
            best_vals = vals_j
            best_edges = edges_j
    
    # Final polish with remaining time
    if best_vals is not None and time.time() < deadline - 4.0:
        remaining = deadline - 4.0 - time.time()
        iters_final = max(3000, int(remaining * 2000))
        
        sc, vals_f = optimize_values_for_edges(best_vals, best_edges, budget_iters=iters_final // 2)
        if sc < best_score:
            best_score = sc
            best_vals = vals_f
        
        if time.time() < deadline - 3.0:
            sc, edges_f = optimize_edges_for_values(best_vals, best_edges, budget_iters=iters_final // 2)
            if sc < best_score:
                best_score = sc
                best_edges = edges_f
        
        if time.time() < deadline - 3.0:
            sc, vals_j, edges_j = joint_search(best_vals, best_edges, budget_iters=iters_final // 3)
            if sc < best_score:
                best_score = sc
                best_vals = vals_j
                best_edges = edges_j
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])

    # Validate and convert
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    assert len(final_values) == 14
    assert len(set(final_values)) == 14
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == 35
    edge_set_check = set()
    for e in final_edges:
        assert isinstance(e, list) and len(e) == 2
        assert isinstance(e[0], int) and not isinstance(e[0], bool)
        assert isinstance(e[1], int) and not isinstance(e[1], bool)
        assert 0 <= e[0] < 14 and 0 <= e[1] < 14
        assert e[0] != e[1]
        key = tuple(sorted(e))
        assert key not in edge_set_check
        edge_set_check.add(key)
    
    return {'values': final_values, 'edges': final_edges}
