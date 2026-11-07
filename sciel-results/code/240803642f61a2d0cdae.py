import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

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

    def fast_score(values, edge_list):
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        return max(len(s), len(p))

    def fast_score_with_counts(values, edge_list):
        vals = values
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
        sc = max(len(s), len(p))
        return sc, len(s), len(p)

    def best_edges_for_values(values, budget):
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)
            sums.add(values[e[0]] + values[e[1]])
            prods.add(values[e[0]] * values[e[1]])
        cur = list(chosen)
        cur_score = fast_score(values, cur)
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_cur = list(cur)
            new_cur[i] = e_in
            new_score = fast_score(values, new_cur)
            if new_score <= cur_score:
                cur = new_cur
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000 and random.random() < 0.01:
                    no_improve = 0
        return cur, cur_score

    def greedy_edges_max_collision(values):
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        def collision_score(e):
            _, _, s, p = e
            score = 0
            if s in sums:
                score += 10
            if p in prods:
                score += 10
            score -= (e[2] + e[3]) / 10000.0
            return score
        
        remaining = list(range(len(all_edges)))
        for _ in range(m):
            if not remaining:
                break
            best_idx = -1
            best_score = float('-inf')
            for idx in remaining:
                e = edge_data[idx]
                s = collision_score(e)
                if s > best_score:
                    best_score = s
                    best_idx = idx
            if best_idx == -1:
                break
            u, v, s, p = edge_data[best_idx]
            chosen.append((u, v))
            chosen_set.add((u, v))
            sums.add(s)
            prods.add(p)
            remaining.remove(best_idx)
        
        return chosen

    def best_values_for_edges(edges, budget):
        vals = random.sample(range(1, 41), n)
        elist = list(edges)
        cur_score = fast_score(vals, elist)
        best = list(vals)
        best_score = cur_score
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            if random.random() < 0.5:
                new_val = random.randint(1, 100)
            else:
                new_val = vals[idx] + random.randint(-15, 15)
            if not (1 <= new_val <= 10000) or new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = fast_score(new_vals, elist)
            if s < cur_score:
                vals = new_vals
                cur_score = s
                no_improve = 0
                if s < best_score:
                    best = list(vals)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 3000:
                    no_improve = 0
        return best, best_score

    def targeted_value_moves(vals, elist, budget):
        cur = list(vals)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(cur, elist)
        best = list(cur)
        best_score = cur_score
        no_improve = 0
        it = 0
        
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            
            neighbors = set()
            for u, v in elist:
                if u == idx:
                    neighbors.add(v)
                elif v == idx:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            existing_sums = set()
            existing_prods = set()
            for u, v in elist:
                if u == idx or v == idx:
                    other = v if u == idx else u
                    existing_sums.add(cur[idx] + cur[other])
                    existing_prods.add(cur[idx] * cur[other])
            
            candidates = set()
            
            for nb in neighbors:
                for s_val in existing_sums:
                    cand = s_val - cur[nb]
                    if 1 <= cand <= 10000 and cand != cur[idx]:
                        candidates.add(cand)
            
            for nb in neighbors:
                for p_val in existing_prods:
                    if cur[nb] != 0 and p_val % cur[nb] == 0:
                        cand = p_val // cur[nb]
                        if 1 <= cand <= 10000 and cand != cur[idx]:
                            candidates.add(cand)
            
            for u, v in elist:
                if u == idx or v == idx:
                    continue
                other_sum = cur[u] + cur[v]
                other_prod = cur[u] * cur[v]
                for nb in neighbors:
                    cand = other_sum - cur[nb]
                    if 1 <= cand <= 10000 and cand != cur[idx]:
                        candidates.add(cand)
                    if cur[nb] != 0 and other_prod % cur[nb] == 0:
                        cand = other_prod // cur[nb]
                        if 1 <= cand <= 10000 and cand != cur[idx]:
                            candidates.add(cand)
            
            if not candidates:
                candidates.add(random.randint(1, 50))
            
            new_val = random.choice(list(candidates))
            if new_val in cur:
                continue
            
            new_vals = list(cur)
            new_vals[idx] = new_val
            s, sc, pc = fast_score_with_counts(new_vals, elist)
            
            if s < cur_score or (s == cur_score and (sc < cur_sc or pc < cur_pc)):
                cur = new_vals
                cur_score, cur_sc, cur_pc = s, sc, pc
                no_improve = 0
                if s < best_score:
                    best = list(cur)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                cur = new_vals
                cur_score, cur_sc, cur_pc = s, sc, pc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000:
                    no_improve = 0
        
        return best, best_score

    def joint_search(vals0, edges0, budget):
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score, cur_sc, cur_pc = fast_score_with_counts(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            r = random.random()
            if r < 0.35:
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                new_elist = list(elist)
                new_elist[i] = e_in
                s, sc, pc = fast_score_with_counts(vals, new_elist)
                if s < cur_score or (s == cur_score and (sc < cur_sc or pc < cur_pc)):
                    elist = new_elist
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score, cur_sc, cur_pc = s, sc, pc
            elif r < 0.65:
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                if s < cur_score or (s == cur_score and (sc < cur_sc or pc < cur_pc)):
                    vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc
            else:
                idx = random.randint(0, n - 1)
                neighbors = [v for u, v in elist if u == idx or v == idx]
                if not neighbors:
                    continue
                existing_sums = set()
                existing_prods = set()
                for u, v in elist:
                    if u == idx or v == idx:
                        other = v if u == idx else u
                        existing_sums.add(vals[idx] + vals[other])
                        existing_prods.add(vals[idx] * vals[other])
                candidates = set()
                for s_val in existing_sums:
                    for nb in neighbors:
                        cand = s_val - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                for p_val in existing_prods:
                    for nb in neighbors:
                        if vals[nb] != 0 and p_val % vals[nb] == 0:
                            cand = p_val // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                for u, v in elist:
                    if u == idx or v == idx:
                        continue
                    other_sum = vals[u] + vals[v]
                    other_prod = vals[u] * vals[v]
                    for nb in neighbors:
                        cand = other_sum - vals[nb]
                        if 1 <= cand <= 10000 and cand != vals[idx]:
                            candidates.add(cand)
                        if vals[nb] != 0 and other_prod % vals[nb] == 0:
                            cand = other_prod // vals[nb]
                            if 1 <= cand <= 10000 and cand != vals[idx]:
                                candidates.add(cand)
                candidates.add(random.randint(1, 50))
                if not candidates:
                    continue
                new_val = random.choice(list(candidates))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s, sc, pc = fast_score_with_counts(new_vals, elist)
                if s < cur_score or (s == cur_score and (sc < cur_sc or pc < cur_pc)):
                    vals = new_vals
                    cur_score, cur_sc, cur_pc = s, sc, pc
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    def try_direct_construction():
        best = None
        pools = [
            list(range(1, 25)),
            list(range(2, 30)),
            list(range(1, 20)),
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
            [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
            [2,3,4,5,6,7,8,9,10,11,12,13,14,16],
            [1,2,3,4,5,6,7,8,9,11,12,14,16,18],
            [1,2,3,4,5,6,7,8,9,10,12,14,16,18],
            [1,2,3,4,5,6,7,8,9,10,11,12,14,16],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28],
            [1,3,5,7,9,11,13,15,17,19,21,23,25,27],
            [1,2,4,8,16,3,6,12,24,5,10,20,40,7],
            # New pools designed for multiplicative structure
            [2,3,4,6,8,9,12,16,18,24,27,32,36,48],
            [1,2,4,8,16,32,3,6,12,24,48,5,10,20],
            [1,2,3,5,7,11,13,17,19,23,29,31,37,41], # primes
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,9,10,11,12,13,16],
        ]
        for pool in pools:
            for _ in range(4):
                vals = random.sample(pool, n)
                edges_greedy = greedy_edges_max_collision(vals)
                sc_greedy = fast_score(vals, edges_greedy)
                
                edges_ls, sc_ls = best_edges_for_values(vals, budget=2000)
                
                if sc_greedy < sc_ls:
                    edges, sc = edges_greedy, sc_greedy
                else:
                    edges, sc = edges_ls, sc_ls
                
                if best is None or sc < best[0]:
                    best = (sc, list(vals), edges)
                if time.time() > deadline - 35:
                    return best
        return best

    def constraint_search(vals, max_score):
        """
        Backtracking search for edges given fixed values and a max score constraint.
        Returns a list of m edges if found, else None.
        """
        sums = set()
        prods = set()
        chosen = []
        chosen_set = set()
        
        # Pre-calculate sums/prods for all edges
        edge_props = {}
        for u, v in all_edges:
            edge_props[(u, v)] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Order edges: prioritize those that add few new sums/prods
        def edge_priority(e):
            s, p = edge_props[e]
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            # Prefer edges that don't increase the score
            # Rough heuristic: count collisions
            coll_s = 0
            coll_p = 0
            # This is expensive to compute dynamically in recursion, so use static proxy
            # For now, just sort by value sum to keep numbers small?
            # Better: sort by how "common" the sum/product is likely to be
            return vals[e[0]] + vals[e[1]]
        
        # Actually, for backtracking, we need to be careful.
        # Let's use a simple recursive backtracker with pruning.
        
        def backtrack(idx, current_sums, current_prods, current_edges):
            if len(current_edges) == m:
                return True
            if time.time() > deadline - 1.0:
                return False
            
            # Pruning: if distinct sums > max_score or distinct prods > max_score, fail
            if len(current_sums) > max_score or len(current_prods) > max_score:
                return False
            
            # Heuristic: pick next edge
            # To make this faster, we can pre-sort edges by some static measure
            # But dynamic state matters.
            # Let's iterate through remaining edges.
            
            # Optimization: if we have few edges left, we can be more aggressive
            remaining_needed = m - len(current_edges)
            
            # Candidate ordering:
            # 1. Edges that reuse existing sums/prods
            # 2. Edges with small sums/prods
            
            candidates = []
            for e in all_edges:
                if e in chosen_set:
                    continue
                s, p = edge_props[e]
                new_s = s not in current_sums
                new_p = p not in current_prods
                # Score: prefer no new, then small
                score = (0 if not new_s else 1, 0 if not new_p else 1, s+p)
                candidates.append((score, e))
            
            candidates.sort(key=lambda x: x[0])
            
            # Limit branching factor to avoid explosion
            # If max_score is tight, we must pick well
            branch_limit = 50 if max_score < 40 else 100
            
            for score, e in candidates[:branch_limit]:
                s, p = edge_props[e]
                chosen_set.add(e)
                new_sums = set(current_sums)
                new_prods = set(current_prods)
                new_sums.add(s)
                new_prods.add(p)
                
                if backtrack(idx + 1, new_sums, new_prods, current_edges + [e]):
                    chosen.extend(current_edges + [e])
                    return True
                
                chosen_set.remove(e)
            
            return False
        
        if backtrack(0, set(), set(), []):
            return chosen
        return None

    def find_optimal_structure():
        """
        Structural search:
        1. Guess a target score K (start low).
        2. Try to find values and edges that fit in K.
        """
        # Start with K = 35 (theoretical min) and increase
        # But 35 is very hard. Let's start higher and go down?
        # Or start with a reasonable guess like 40-50 and refine.
        
        # Let's try to construct a solution with K=45 first, then try to lower it.
        # Or use the previous best as a starting point for K.
        
        # Strategy:
        # Iterate K from 35 to 60.
        # For each K, try a few random value sets and run constraint_search.
        
        # To be efficient, we only try this for a few K values and a few value sets.
        
        best_result = None
        
        # We can't try all K. Let's assume the previous best was around 40-50.
        # Let's try K in [35, 36, 37, 38, 39, 40, 42, 45, 50]
        # But constraint search is slow.
        
        # Alternative: Use the previous best score as a hint.
        # The current program found ~48 (score 2.07 implies value ~48 if seed=100? No, score=seed/value.
        # If score is 2.0769, and seed is likely 100 or similar?
        # Wait, "search score is seed/value".
        # If score is 2.0769, and value is V, then seed/V = 2.0769.
        # If seed=100, V=48.
        # If seed=50, V=24.
        # Let's assume the value is around 40-50.
        
        # Let's try to improve upon the current best by using constraint search with K = current_best - 1, -2, etc.
        # But we don't know current_best until we run the other phases.
        
        # So, let's run the standard heuristic search first to get a baseline,
        # THEN use constraint search to try to beat it.
        
        return None # Placeholder, will be called later

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Direct Construction (Heuristic)
    d = try_direct_construction()
    if d is not None:
        best_score, best_vals, best_edges = d

    # Phase 2: Joint Search
    random.seed(1234)
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), set(best_edges)))
    for _ in range(10):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    remaining = deadline - time.time() - 10.0
    per = max(3000, int(remaining / len(starts) / 0.00005))
    for vals0, edges0 in starts:
        if time.time() > deadline - 10.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges

    # Phase 3: Constraint-Based Refinement
    # Try to lower the score by 1 or 2 using backtracking on edges for fixed values.
    if time.time() < deadline - 15.0 and best_vals is not None:
        # Try K = best_score - 1, best_score - 2
        for k_offset in [1, 2, 3]:
            target_k = best_score - k_offset
            if target_k < 35:
                continue
            
            # Try a few variations of the current best values
            # 1. Current best values
            # 2. Slightly perturbed values
            
            attempts = 0
            max_attempts = 5
            found = False
            
            while attempts < max_attempts and not found and time.time() < deadline - 15.0:
                attempts += 1
                
                if attempts == 1:
                    v_trial = list(best_vals)
                else:
                    # Perturb
                    v_trial = list(best_vals)
                    idx = random.randint(0, n-1)
                    v_trial[idx] = random.randint(1, 100)
                    # Ensure distinct
                    if len(set(v_trial)) < n:
                        continue

                edges_found = constraint_search(v_trial, target_k)
                if edges_found is not None:
                    # Verify
                    sc, sc_s, sc_p = fast_score_with_counts(v_trial, edges_found)
                    if sc <= target_k:
                        best_score = sc
                        best_vals = v_trial
                        best_edges = set(edges_found)
                        found = True
                        # If we found a lower score, we can break and try even lower
                        break

    # Phase 4: Polish
    if time.time() < deadline - 5.0 and best_vals is not None:
        elist = list(best_edges)
        it = 0
        while it < 25000 and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            new_elist = list(elist)
            new_elist[i] = e_in
            s = fast_score(best_vals, new_elist)
            if s < best_score:
                elist = new_elist
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    if time.time() < deadline - 2.0 and best_vals is not None:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 20000 and time.time() < deadline - 1.0:
            it += 1
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, 100)
            if new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = fast_score(new_vals, elist)
            if s < best_score:
                vals = new_vals
                best_score = s
                best_vals = new_vals

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
