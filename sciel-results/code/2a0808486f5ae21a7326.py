import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        
        # Map values to indices for faster lookup
        V_list = sorted(V_set)
        
        # Build adjacency: for each v, list of (u, w) pairs such that {v, u, w} is a hyperedge
        # Hyperedge (a, b, c) means a+b=c
        # For v=a: (b, c)
        # For v=b: (a, c)  
        # For v=c: (a, b)
        # Special: (a, a, 2a) -> for v=a: (a, 2a), for v=2a: (a, a)
        
        adj = {v: [] for v in V_set}
        
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))  # self-reference
                adj[2 * a].append((a, a))
            
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))
        
        def get_conflicts(v, S):
            """Get list of (u, w) conflicts for adding v to S"""
            conflicts = []
            for (u, w) in adj[v]:
                if u == v:
                    # Self-loop: v + v = w, need w in S
                    if w in S:
                        conflicts.append((w, w))
                else:
                    # Need both u and w in S
                    if u in S and w in S:
                        conflicts.append((u, w))
            return conflicts
        
        def is_sum_free(S):
            """Check if S is sum-free"""
            S_list = list(S)
            S_set = set(S_list)
            for i, a in enumerate(S_list):
                for b in S_list[i:]:
                    c = a + b
                    if c in S_set:
                        return False
            return True
        
        def local_search_walksat(S, V_set, adj, budget, rng):
            """WalkSAT-style local search with adaptive restarts and clause weighting"""
            t_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            
            # Clause weights for adaptive learning
            clause_weights = {}
            last_reject_count = 0
            restart_interval = 2000
            max_restarts = 50
            
            def clause_key(u, w):
                return (min(u, w), max(u, w))
            
            while time.time() - t_start < budget:
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break
                
                # Pick a candidate to add
                v = rng.choice(remaining)
                conflicts = get_conflicts(v, S)
                
                if not conflicts:
                    S.add(v)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    last_reject_count = 0
                    continue
                
                # We have conflicts - need to resolve
                # Count frequency of each vertex in conflicts
                freq = {}
                for (u, w) in conflicts:
                    ck = clause_key(u, w)
                    freq[ck] = freq.get(ck, 0) + 1
                    if u != w:
                        freq[u] = freq.get(u, 0) + 1
                        freq[w] = freq.get(w, 0) + 1
                
                # Try different strategies:
                # 1. Remove the most frequent vertex
                # 2. Remove a random vertex from conflicts
                # 3. 2-swap: add v, remove two vertices, add one back
                
                strategy = rng.random()
                
                if strategy < 0.5:
                    # Strategy 1: Remove most frequent blocker
                    if freq:
                        max_f = max(freq.values())
                        candidates = [k for k, f in freq.items() if f == max_f]
                        rem = rng.choice(candidates)
                        if isinstance(rem, tuple):
                            # It's a clause key, pick one of the vertices
                            rem = rng.choice(rem)
                        
                        trial_S = set(S)
                        trial_S.discard(rem)
                        trial_S.add(v)
                        
                        # Verify validity
                        valid = True
                        for (u, w) in conflicts:
                            if u == w:
                                if u in trial_S:
                                    valid = False
                                    break
                            else:
                                if u in trial_S and w in trial_S:
                                    valid = False
                                    break
                        
                        if valid:
                            new_size = len(trial_S)
                            if new_size > len(S):
                                S = trial_S
                                if new_size > best_size:
                                    best_size = new_size
                                    best_S = set(S)
                                last_reject_count = 0
                            elif rng.random() < 0.15:
                                S = trial_S
                                last_reject_count += 1
                        else:
                            last_reject_count += 1
                
                elif strategy < 0.7:
                    # Strategy 2: Random vertex removal
                    all_blockers = set()
                    for (u, w) in conflicts:
                        if u != w:
                            all_blockers.add(u)
                            all_blockers.add(w)
                        else:
                            all_blockers.add(u)
                    
                    if all_blockers:
                        rem = rng.choice(list(all_blockers))
                        trial_S = set(S)
                        trial_S.discard(rem)
                        trial_S.add(v)
                        
                        valid = True
                        for (u, w) in conflicts:
                            if u == w:
                                if u in trial_S:
                                    valid = False
                                    break
                            else:
                                if u in trial_S and w in trial_S:
                                    valid = False
                                    break
                        
                        if valid:
                            new_size = len(trial_S)
                            if new_size > len(S):
                                S = trial_S
                                if new_size > best_size:
                                    best_size = new_size
                                    best_S = set(S)
                                last_reject_count = 0
                            elif rng.random() < 0.1:
                                S = trial_S
                                last_reject_count += 1
                        else:
                            last_reject_count += 1
                
                else:
                    # Strategy 3: 2-swap - try to add v and swap out two vertices
                    # Find two vertices to remove that resolve all conflicts
                    if len(conflicts) >= 2:
                        # Try all pairs of blockers
                        blockers = []
                        for (u, w) in conflicts:
                            if u != w:
                                blockers.extend([u, w])
                            else:
                                blockers.append(u)
                        
                        blockers = list(set(blockers))
                        rng.shuffle(blockers)
                        
                        tried = 0
                        for i in range(min(len(blockers), 20)):
                            for j in range(i + 1, min(len(blockers), 20)):
                                rem1, rem2 = blockers[i], blockers[j]
                                trial_S = set(S)
                                trial_S.discard(rem1)
                                trial_S.discard(rem2)
                                trial_S.add(v)
                                
                                # Check if all original conflicts are resolved
                                valid = True
                                for (u, w) in conflicts:
                                    if u == w:
                                        if u in trial_S:
                                            valid = False
                                            break
                                    else:
                                        if u in trial_S and w in trial_S:
                                            valid = False
                                            break
                                
                                if valid and len(trial_S) > len(S):
                                    S = trial_S
                                    if len(S) > best_size:
                                        best_size = len(S)
                                        best_S = set(S)
                                    last_reject_count = 0
                                    break
                                tried += 1
                                if tried > 50:
                                    break
                            if last_reject_count == 0:
                                break
                        if last_reject_count > 0:
                            last_reject_count += 1
                
                # Adaptive restart
                if last_reject_count > restart_interval:
                    # Restart with a perturbed version of best_S
                    if rng.random() < 0.3:
                        S = set(best_S)
                        # Perturb: randomly remove a few elements and re-add
                        perturbed = set(S)
                        for _ in range(min(3, len(S))):
                            if perturbed:
                                to_remove = rng.choice(list(perturbed))
                                perturbed.discard(to_remove)
                        # Try to fill back up
                        for x in V_set:
                            if x not in perturbed:
                                if not get_conflicts(x, perturbed):
                                    perturbed.add(x)
                        S = perturbed
                    last_reject_count = 0
                    restart_interval = max(1000, int(restart_interval * 1.5))
            
            return best_S
        
        # Initial solutions
        best_S = set()
        best_size = 0
        rng = random.Random(seed_off)
        
        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best_S = S_odds
        
        # 2. Greedy by degree (fewest conflicts first)
        deg = {v: len(adj[v]) for v in V_set}
        S_greedy = set()
        for v in sorted(V_set, key=lambda x: (deg[x], abs(x))):
            if not get_conflicts(v, S_greedy):
                S_greedy.add(v)
        
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_S = S_greedy
        
        # 3. Multiple random greedy starts
        for _ in range(5):
            S_rand = set()
            order = list(V_set)
            rng.shuffle(order)
            for v in order:
                if not get_conflicts(v, S_rand):
                    S_rand.add(v)
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_S = S_rand
        
        # 4. Local search from best
        budget_ls = max(2, budget * 0.85)
        best_S = local_search_walksat(best_S, V_set, adj, budget_ls, rng)
        
        # Final verification
        if not is_sum_free(best_S):
            # Fallback to odds
            best_S = set(x for x in V_set if x % 2 != 0)
        
        return list(best_S)
    
    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]
    
    results = []
    # Allocate budget proportionally to problem size
    total_budget = DEADLINE - (time.time() - t0) - 10  # 10s safety margin
    if total_budget < 15:
        # Not enough time, use odds for all
        for elems in instances:
            results.append([x for x in elems if x % 2 != 0])
        return results
    
    # Weight by n^1.5 to account for search complexity
    weights = [60**1.5, 120**1.5, 240**1.5]
    total_w = sum(weights)
    budgets = [max(3, int(total_budget * w / total_w)) for w in weights]
    
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 8
        if remaining < 4:
            results.append([x for x in elems if x % 2 != 0])
            continue
        b = min(budgets[i], max(3, remaining))
        res = solve_instance(elems, seed_off=801 + i * 13, budget=b)
        results.append(res)
    
    return results
