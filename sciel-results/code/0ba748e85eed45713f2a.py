import time
import random
import math
import sys

def _precompute(n=180):
    """Precompute forbidden triples and adjacency pairs."""
    triples = []
    for x in range(1, n + 1):
        x2 = x * x
        d = 1
        while d * d <= x2:
            if x2 % d == 0:
                e = x2 // d
                if d < e:
                    y = x + d
                    z = x + e
                    if y <= n and z <= n and y != z:
                        triples.append((x - 1, y - 1, z - 1))
            d += 1
    # For each vertex, list of (other1, other2) pairs it participates in
    pairs = [[] for _ in range(n)]
    for (i, j, k) in triples:
        pairs[i].append((j, k))
        pairs[j].append((i, k))
        pairs[k].append((i, j))
    return triples, pairs

def _min_degree_greedy(triples, pairs, n, deadline, seed_val=42):
    """
    Minimum-degree greedy: iteratively select the vertex with minimum
    conflict degree among those not yet excluded, building up the set.
    This exploits structural asymmetry in the hypergraph.
    """
    random.seed(seed_val)
    S = [0] * n
    in_set = [False] * n
    # Track which vertices are still "available" (not excluded)
    available = [True] * n
    
    # Conflict count: for each vertex, how many forbidden triples would be
    # completed if we added it (given current S)
    # We'll recompute incrementally
    
    # Start by trying to add all vertices in an order determined by
    # minimum current conflict degree
    order = list(range(n))
    random.shuffle(order)
    
    # First pass: greedy add in shuffled order, but prioritize low-conflict
    for v in order:
        if time.time() > deadline:
            break
        if not available[v]:
            continue
        # Check if adding v creates any violation
        can_add = True
        for (u, w) in pairs[v]:
            if S[u] and S[w]:
                can_add = False
                break
        if can_add:
            S[v] = 1
            in_set[v] = True
    
    # Second pass: try to add remaining vertices, prioritizing those with
    # fewest conflicts with current set
    remaining = [v for v in range(n) if not S[v]]
    
    # Sort remaining by number of active conflicts (pairs where both others are in S)
    def conflict_count(v):
        c = 0
        for (u, w) in pairs[v]:
            if S[u] and S[w]:
                c += 1
        return c
    
    remaining.sort(key=conflict_count)
    
    for v in remaining:
        if time.time() > deadline:
            break
        if S[v]:
            continue
        can_add = True
        for (u, w) in pairs[v]:
            if S[u] and S[w]:
                can_add = False
                break
        if can_add:
            S[v] = 1
            in_set[v] = True
    
    return S, sum(S)

def _sa_refine(S, triples, pairs, n, t_start, deadline, cur_v, cur_size, best, best_size):
    """Simulated annealing refinement with single-flip and 2-swap moves."""
    T = 1.0
    step = 0
    while time.time() < deadline:
        step += 1
        move_type = random.random()
        
        if move_type < 0.7:
            # Single flip move
            v = random.randrange(n)
            old = S[v]
            if old == 0:
                addc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        addc += 1
                new_v = cur_v + addc
                new_size = cur_size + 1
            else:
                remc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        remc += 1
                new_v = cur_v - remc
                new_size = cur_size - 1
            
            new_score = new_size - 1000.0 * new_v
            cur_score = cur_size - 1000.0 * cur_v
            dE = new_score - cur_score
            
            if dE >= 0 or random.random() < math.exp(dE / T):
                S[v] = 1 - old
                cur_v = new_v
                cur_size = new_size
                if new_v == 0 and cur_size > best_size:
                    best[:] = S[:]
                    best_size = cur_size
        else:
            # 2-swap move: remove one element and add another
            # Pick a random element to remove (if in set) and a random element to add (if not in set)
            v_remove = random.randrange(n)
            v_add = random.randrange(n)
            
            if S[v_remove] == 0 or S[v_add] == 1 or v_remove == v_add:
                # Invalid swap, try single flip instead
                v = random.randrange(n)
                old = S[v]
                if old == 0:
                    addc = 0
                    for (u, w) in pairs[v]:
                        if S[u] and S[w]:
                            addc += 1
                    new_v = cur_v + addc
                    new_size = cur_size + 1
                else:
                    remc = 0
                    for (u, w) in pairs[v]:
                        if S[u] and S[w]:
                            remc += 1
                    new_v = cur_v - remc
                    new_size = cur_size - 1
                
                new_score = new_size - 1000.0 * new_v
                cur_score = cur_size - 1000.0 * cur_v
                dE = new_score - cur_score
                
                if dE >= 0 or random.random() < math.exp(dE / T):
                    S[v] = 1 - old
                    cur_v = new_v
                    cur_size = new_size
                    if new_v == 0 and cur_size > best_size:
                        best[:] = S[:]
                        best_size = cur_size
            else:
                # Perform 2-swap: remove v_remove, add v_add
                # Compute delta in violations and size
                # Removing v_remove: decreases violations where v_remove is in a triple with two other in-set elements
                rem_viol = 0
                for (u, w) in pairs[v_remove]:
                    if S[u] and S[w]:
                        rem_viol += 1
                
                # Adding v_add: increases violations where v_add is in a triple with two other in-set elements
                # But we need to account for the removal of v_remove
                add_viol = 0
                for (u, w) in pairs[v_add]:
                    if S[u] and S[w]:
                        if u == v_remove or w == v_remove:
                            # This violation would be broken by removing v_remove
                            pass
                        else:
                            add_viol += 1
                
                new_v = cur_v - rem_viol + add_viol
                new_size = cur_size  # Remove one, add one: size unchanged
                
                new_score = new_size - 1000.0 * new_v
                cur_score = cur_size - 1000.0 * cur_v
                dE = new_score - cur_score
                
                if dE >= 0 or random.random() < math.exp(dE / T):
                    S[v_remove] = 0
                    S[v_add] = 1
                    cur_v = new_v
                    cur_size = new_size
                    if new_v == 0 and cur_size > best_size:
                        best[:] = S[:]
                        best_size = cur_size
        
        T *= 0.99999
        if T < 0.01:
            T = 0.01
    
    return S, cur_v, cur_size, best, best_size

def _greedy_add(S, pairs, n):
    """Try to add all possible elements greedily."""
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if not S[v]:
                ok = True
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        ok = False
                        break
                if ok:
                    S[v] = 1
                    improved = True
    return S, sum(S)

def _repair(S, triples, pairs, n):
    """Remove elements to fix violations."""
    S = S[:]
    changed = True
    while changed:
        changed = False
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                # Remove the element with highest degree
                di = len(pairs[i])
                dj = len(pairs[j])
                dk = len(pairs[k])
                if di >= dj and di >= dk:
                    S[i] = 0
                elif dj >= di and dj >= dk:
                    S[j] = 0
                else:
                    S[k] = 0
                changed = True
                break
    return S, sum(S)

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin before 90s kill
    n = 180
    
    triples, pairs = _precompute(n)
    
    best = [0] * n
    best_size = 0
    
    # Phase 1: Multiple minimum-degree greedy constructions with different seeds
    # This is the structural change: exploit hypergraph sparsity via min-degree selection
    num_greedy_trials = 8
    greedy_time_budget = (DEADLINE - (time.time() - t0)) * 0.3
    per_greedy_time = greedy_time_budget / num_greedy_trials
    
    for trial in range(num_greedy_trials):
        if time.time() - t0 > DEADLINE:
            break
        
        # Different strategies for each trial
        if trial == 0:
            # Standard min-degree greedy
            S, cur_size = _min_degree_greedy(triples, pairs, n, time.time() + per_greedy_time, seed_val=42)
        elif trial == 1:
            # Min-degree greedy with different seed
            S, cur_size = _min_degree_greedy(triples, pairs, n, time.time() + per_greedy_time, seed_val=123)
        elif trial == 2:
            # Start from high interval, then min-degree add
            S = [0] * n
            for i in range(90, 180):
                S[i] = 1
            # Greedy add remaining
            S, cur_size = _greedy_add(S, pairs, n)
        elif trial == 3:
            # Start from medium interval
            S = [0] * n
            for i in range(60, 180):
                S[i] = 1
            S, cur_size = _repair(S, triples, pairs, n)
            S, cur_size = _greedy_add(S, pairs, n)
        elif trial == 4:
            # Random 50%
            random.seed(777)
            S = [0] * n
            for i in range(n):
                if random.random() < 0.5:
                    S[i] = 1
            S, cur_size = _repair(S, triples, pairs, n)
            S, cur_size = _greedy_add(S, pairs, n)
        elif trial == 5:
            # Random 40%
            random.seed(888)
            S = [0] * n
            for i in range(n):
                if random.random() < 0.4:
                    S[i] = 1
            S, cur_size = _repair(S, triples, pairs, n)
            S, cur_size = _greedy_add(S, pairs, n)
        elif trial == 6:
            # Min-degree greedy, different seed
            S, cur_size = _min_degree_greedy(triples, pairs, n, time.time() + per_greedy_time, seed_val=999)
        else:
            # Start from high interval [75, 180]
            S = [0] * n
            for i in range(75, 180):
                S[i] = 1
            S, cur_size = _repair(S, triples, pairs, n)
            S, cur_size = _greedy_add(S, pairs, n)
        
        # Check feasibility and update best
        feasible = True
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                feasible = False
                break
        if feasible and cur_size > best_size:
            best[:] = S[:]
            best_size = cur_size
    
    # Phase 2: SA refinement from the best solution found
    if time.time() - t0 < DEADLINE:
        S = best[:]
        cur_size = sum(S)
        cur_v = 0
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                cur_v += 1
        
        # SA from best
        sa_deadline = min(time.time() + (DEADLINE - (time.time() - t0)) * 0.5, DEADLINE)
        S, cur_v, cur_size, best, best_size = _sa_refine(
            S, triples, pairs, n, t0, sa_deadline, cur_v, cur_size, best, best_size
        )
        
        # Greedy add after SA
        S, cur_size = _greedy_add(S, pairs, n)
        feasible = True
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                feasible = False
                break
        if feasible and cur_size > best_size:
            best[:] = S[:]
            best_size = cur_size
        
        # Phase 2b: Second SA pass with lower initial temperature for polishing
        if time.time() - t0 < DEADLINE:
            S2 = best[:]
            cur_size2 = sum(S2)
            cur_v2 = 0
            for (i, j, k) in triples:
                if S2[i] and S2[j] and S2[k]:
                    cur_v2 += 1
            
            sa2_deadline = min(time.time() + (DEADLINE - (time.time() - t0)) * 0.3, DEADLINE)
            S2, cur_v2, cur_size2, best, best_size = _sa_refine(
                S2, triples, pairs, n, t0, sa2_deadline, cur_v2, cur_size2, best, best_size
            )
            
            S2, cur_size2 = _greedy_add(S2, pairs, n)
            feasible2 = True
            for (i, j, k) in triples:
                if S2[i] and S2[j] and S2[k]:
                    feasible2 = False
                    break
            if feasible2 and cur_size2 > best_size:
                best[:] = S2[:]
                best_size = cur_size2
    
    # Phase 3: Final greedy add from best
    if time.time() - t0 < DEADLINE:
        Sb = best[:]
        Sz = sum(Sb)
        Sb, Sz = _greedy_add(Sb, pairs, n)
        if Sz > best_size:
            feasible = True
            for (i, j, k) in triples:
                if Sb[i] and Sb[j] and Sb[k]:
                    feasible = False
                    break
            if feasible:
                best = Sb
                best_size = Sz
    
    out = [i + 1 for i in range(n) if best[i]]
    if not out:
        out = [180]
    return out
