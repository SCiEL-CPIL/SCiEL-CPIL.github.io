import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def build_conflict_graph(V_set):
        """Build adjacency: for each v, list of (u, w) such that {v,u,w} forms a sum-free violation."""
        adj = {v: [] for v in V_set}
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))
        return adj

    def conflict_count_fast(v, S, adj):
        """Count how many hyperedges involving v would be violated if v were in S."""
        count = 0
        for (u, w) in adj[v]:
            if u == v:
                if w in S:
                    count += 1
            else:
                if u in S and w in S:
                    count += 1
        return count

    def greedy_from_order(V_set, order, adj):
        """Greedily add elements in the given order if they don't conflict."""
        S = set()
        for v in order:
            if v in S:
                continue
            can_add = True
            for (u, w) in adj[v]:
                if u == v:
                    if w in S:
                        can_add = False
                        break
                else:
                    if u in S and w in S:
                        can_add = False
                        break
            if can_add:
                S.add(v)
        return S

    def local_search(S, V_set, adj, budget, rng):
        """Local search with add, remove, and swap moves using conflict-count tracking."""
        t_start = time.time()
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        # Precompute conflict counts for all elements
        conf = {v: conflict_count_fast(v, S, adj) for v in V_set if v not in S}
        in_S = set(S)
        
        while time.time() - t_start < budget:
            remaining = [v for v in V_set if v not in in_S]
            if not remaining:
                break
            
            # Try to add a zero-conflict element
            zero_conf = [v for v in remaining if conf.get(v, 0) == 0]
            if zero_conf:
                v = rng.choice(zero_conf)
                S.add(v)
                in_S.add(v)
                # Update conflict counts for other elements
                for u in V_set:
                    if u not in in_S:
                        old_c = conf.get(u, 0)
                        new_c = old_c
                        for (a, b) in adj[u]:
                            if a == u:
                                if b == v and v in in_S:
                                    new_c += 1
                            else:
                                if (a == v or b == v) and (a in in_S and b in in_S):
                                    new_c += 1
                        conf[u] = new_c
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue
            
            # No zero-conflict additions. Try swaps: remove one from S, add one from outside.
            # For each v outside with low conflict, find a u in S that blocks it.
            # Pick v with minimum conflict count.
            min_conf = min(conf.values()) if conf else 0
            if min_conf == 0:
                # Shouldn't happen due to zero_conf check above, but safety
                break
            
            candidates = [v for v in remaining if conf.get(v, 0) == min_conf]
            if not candidates:
                break
            v = rng.choice(candidates)
            
            # Find blockers for v
            blockers = set()
            for (u, w) in adj[v]:
                if u == v:
                    if w in in_S:
                        blockers.add(w)
                else:
                    if u in in_S and w in in_S:
                        blockers.add(u)
                        blockers.add(w)
            
            if not blockers:
                # Shouldn't happen if conf > 0, but safety
                continue
            
            # Try removing each blocker and adding v
            best_swap = None
            best_swap_net = -1
            for rem in list(blockers):
                # Simulate: remove rem, add v
                trial_in_S = set(in_S)
                trial_in_S.discard(rem)
                trial_in_S.add(v)
                
                # Check if valid
                valid = True
                for (u, w) in adj[v]:
                    if u == v:
                        if w in trial_in_S:
                            valid = False
                            break
                    else:
                        if u in trial_in_S and w in trial_in_S:
                            valid = False
                            break
                
                if valid:
                    # Compute net change in total conflicts (approximate)
                    # Removing rem may reduce conflicts for other elements
                    # Adding v adds conflicts
                    # For simplicity, accept if size doesn't decrease
                    # Size: -1 + 1 = 0, so neutral
                    # Accept with some probability to explore
                    if rng.random() < 0.3:
                        best_swap = rem
                        best_swap_net = 0
                        break
            
            if best_swap is not None:
                S.discard(best_swap)
                in_S.discard(best_swap)
                S.add(v)
                in_S.add(v)
                # Update conflict counts
                for u in V_set:
                    if u not in in_S:
                        old_c = conf.get(u, 0)
                        new_c = old_c
                        for (a, b) in adj[u]:
                            if a == u:
                                if b == v and v in in_S:
                                    new_c += 1
                                elif b == best_swap and best_swap not in in_S:
                                    new_c -= 1
                            else:
                                if (a == v or b == v) and (a in in_S and b in in_S):
                                    new_c += 1
                                elif (a == best_swap or b == best_swap) and (a in in_S and b in in_S):
                                    new_c -= 1
                        conf[u] = max(0, new_c)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
            
            # Periodically try random removals to escape local optima
            if rng.random() < 0.05:
                removable = [v for v in in_S if len([x for (a, b) in adj[v] if (a == v and b in in_S) or (a in in_S and b in in_S)]) <= 1]
                if removable:
                    rem = rng.choice(removable)
                    S.discard(rem)
                    in_S.discard(rem)
                    # Update conflict counts
                    for u in V_set:
                        if u not in in_S:
                            old_c = conf.get(u, 0)
                            new_c = old_c
                            for (a, b) in adj[u]:
                                if a == u:
                                    if b == rem and rem not in in_S:
                                        new_c -= 1
                                else:
                                    if (a == rem or b == rem) and (a in in_S and b in in_S):
                                        new_c -= 1
                            conf[u] = max(0, new_c)
        
        return best_S

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V_set = set(elements)
        n = len(V_set)
        
        adj = build_conflict_graph(V_set)
        rng = random.Random(seed_off)
        
        best_S = set()
        best_size = 0
        
        # Multiple restarts with different initializations
        n_restarts = max(1, int(budget / 8))
        per_restart = budget / n_restarts
        
        for restart in range(n_restarts):
            if time.time() - t_start > budget:
                break
            
            # Different initial strategies per restart
            if restart == 0:
                # Odds
                S_init = set(x for x in V_set if x % 2 != 0)
            elif restart == 1:
                # Greedy by degree
                deg = {v: len(adj[v]) for v in V_set}
                order = sorted(V_set, key=lambda x: (deg[x], abs(x)))
                S_init = greedy_from_order(V_set, order, adj)
            elif restart == 2:
                # Random greedy
                order = list(V_set)
                rng.shuffle(order)
                S_init = greedy_from_order(V_set, order, adj)
            else:
                # Perturb best_S
                if best_S:
                    S_init = set(best_S)
                    # Randomly remove some elements
                    to_remove = rng.sample(list(S_init), min(len(S_init) // 3, 10))
                    for v in to_remove:
                        S_init.discard(v)
                else:
                    S_init = set(x for x in V_set if x % 2 != 0)
            
            if len(S_init) > best_size:
                best_size = len(S_init)
                best_S = set(S_init)
            
            # Local search from this initialization
            remaining_budget = max(1, budget - (time.time() - t_start))
            ls_budget = min(per_restart, remaining_budget)
            if ls_budget > 0.5:
                S_ls = local_search(S_init, V_set, adj, ls_budget, rng)
                if len(S_ls) > best_size:
                    best_size = len(S_ls)
                    best_S = set(S_ls)
        
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
