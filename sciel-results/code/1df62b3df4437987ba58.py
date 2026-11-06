import math
import time
import random

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill

    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]

    # Conflicts restricted to the candidate set.
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    # Fast coloring upper bound for independent set on a subgraph.
    def color_bound(nodes):
        if not nodes:
            return 0
        # Order by degree descending for better coloring
        degs = [(len(conflict[u] & nodes), u) for u in nodes]
        degs.sort(reverse=True)
        ordered = [u for _, u in degs]
        
        color_of = {}
        next_color = 0
        for u in ordered:
            used = set()
            for v in conflict[u]:
                if v in nodes and v in color_of:
                    used.add(color_of[v])
            c = 0
            while c in used:
                c += 1
            color_of[u] = c
            if c > next_color:
                next_color = c
        return next_color + 1

    def time_up():
        return time.time() - start > deadline - 3.0

    # Branch-and-bound for maximum independent set
    def branch_and_bound(pool, time_budget):
        pool = set(pool)
        if not pool:
            return set()
        
        # Greedy initial solution
        rem = set(pool)
        greedy = []
        while rem:
            best = None
            bd = None
            for node in rem:
                d = len(conflict[node] & rem)
                if bd is None or d < bd:
                    bd, best = d, node
                    if d == 0:
                        break
            greedy.append(best)
            rem.discard(best)
            rem -= conflict[best]
        
        best = set(greedy)
        blen = len(best)
        
        def search(sub, fixed):
            nonlocal best, blen
            if not sub:
                if len(fixed) > blen:
                    blen, best = len(fixed), set(fixed)
                return
            if time.time() - start > time_budget:
                return
            ub = len(fixed) + color_bound(sub)
            if ub <= blen:
                return
            # Pick high-degree vertex
            v = None
            hv = -1
            for u in sub:
                d = len(conflict[u] & sub)
                if d > hv:
                    hv, v = d, u
            # Branch include
            sub_inc = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + color_bound(sub_inc) > blen:
                search(sub_inc, fixed | {v})
            # Branch exclude
            sub_exc = sub - {v}
            if len(fixed) + color_bound(sub_exc) > blen:
                search(sub_exc, fixed)
        
        search(pool, set())
        return best

    # Local search with add/remove/swap moves
    def local_search(initial, time_budget):
        current = set(initial)
        best = set(current)
        blen = len(current)
        
        def try_add(s):
            for a in candidates:
                if a not in s and not (conflict[a] & s):
                    return a
            return None
        
        def try_remove_swap(s):
            # Try to remove one and add two
            for a in list(s):
                s_without = s - {a}
                # Try to add two non-conflicting vertices
                addable = [b for b in candidates if b not in s_without and not (conflict[b] & s_without)]
                if len(addable) >= 2:
                    # Try all pairs
                    for i in range(len(addable)):
                        for j in range(i + 1, len(addable)):
                            b1, b2 = addable[i], addable[j]
                            if b1 != b2 and b1 + b2 not in squares:
                                new_s = s_without | {b1, b2}
                                return new_s, True
                # Try to add one
                if addable:
                    new_s = s_without | {addable[0]}
                    return new_s, True
            return None, False
        
        while time.time() - start < time_budget:
            improved = False
            # Try adds
            a = try_add(current)
            if a is not None:
                current.add(a)
                improved = True
                if len(current) > blen:
                    blen = len(current)
                    best = set(current)
                continue
            
            # Try swap moves
            new_s, did_swap = try_remove_swap(current)
            if new_s is not None and did_swap:
                current = new_s
                improved = True
                if len(current) > blen:
                    blen = len(current)
                    best = set(current)
                continue
            
            # Perturbation: remove a random vertex
            if current:
                victim = random.choice(list(current))
                current.discard(victim)
                improved = True
        
        return best

    # Main algorithm
    best_solution = set()
    
    # 1. Branch-and-bound on full candidate pool
    sol = branch_and_bound(candidates, deadline - 10.0)
    best_solution = set(sol)
    
    # 2. Local search starting from B&B result
    if time.time() - start < deadline - 5.0:
        ls_result = local_search(best_solution, deadline - 5.0)
        if len(ls_result) > len(best_solution):
            best_solution = ls_result
    
    # 3. Multiple random restarts with local search
    remaining_time = deadline - 5.0 - (time.time() - start)
    if remaining_time > 2.0:
        for _ in range(50):
            if time.time() - start > deadline - 5.0:
                break
            # Random initial solution
            rem = set(candidates)
            random_init = []
            while rem and time.time() - start < deadline - 5.0:
                # Pick random vertex, prefer low degree
                weights = [1.0 / (1 + len(conflict[node] & rem)) for node in rem]
                total = sum(weights)
                r = random.random() * total
                cum = 0
                chosen = None
                for node in rem:
                    cum += weights[node]
                    if cum >= r:
                        chosen = node
                        break
                if chosen is None:
                    chosen = random.choice(list(rem))
                random_init.append(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
            
            # Local search from random init
            ls_result = local_search(set(random_init), min(deadline - 5.0, time.time() - start + 2.0))
            if len(ls_result) > len(best_solution):
                best_solution = ls_result
    
    # 4. Final augmentation: add any non-conflicting vertices
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            best_solution.add(a)
    
    # 5. Safety verification
    ok = True
    lst = sorted(best_solution)
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    if not ok:
        # Fallback: deterministic min-degree greedy
        rem = set(candidates)
        fb = []
        while rem:
            mn = None
            md = None
            for node in rem:
                d = len(conflict[node] & rem)
                if md is None or d < md:
                    md, mn = d, node
            fb.append(mn)
            rem.discard(mn)
            rem -= conflict[mn]
        best_solution = set(fb)
    
    return sorted(best_solution)
