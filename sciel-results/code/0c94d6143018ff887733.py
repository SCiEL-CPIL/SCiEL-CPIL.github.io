"""Rewritten program: greedy multi-start + local-search for large sum-free subsets."""
import time

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # wall-clock limit; keep margin
    MARGIN = 2.0      # safety margin seconds

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_build(S, order):
        """Greedy: add elements in given order if sum-free is maintained."""
        chosen = []
        chosen_set = set()
        for idx in order:
            x = S[idx]
            ok = True
            for a in chosen_set:
                if (x - a) in chosen_set:
                    ok = False
                    break
                if a + x in chosen_set:
                    ok = False
                    break
            if ok and 2 * x in chosen_set:
                ok = False
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def local_search_swap(S, initial):
        """Local search with swap moves: remove one element, try to add one (net size same or +1).
        Also try pure additions. Returns improved set."""
        current = list(initial)
        current_set = set(current)
        best = len(current)
        n = len(S)
        
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() - start_time > DEADLINE - MARGIN:
                break
                
            # First, try pure additions (no removal)
            for y in S:
                if y in current_set:
                    continue
                ok = True
                for a in current_set:
                    if (y - a) in current_set:
                        ok = False
                        break
                    if a + y in current_set:
                        ok = False
                        break
                if ok and 2 * y in current_set:
                    ok = False
                if ok:
                    current.append(y)
                    current_set.add(y)
                    best = len(current)
                    improved = True
            
            # Then, try swap moves: remove x, add y
            if not improved:
                for x in current:
                    # Try removing x
                    test_set = set(current_set)
                    test_set.discard(x)
                    for y in S:
                        if y in test_set:
                            continue
                        # Check if test_set ∪ {y} is sum-free
                        ok = True
                        for a in test_set:
                            if (y - a) in test_set:
                                ok = False
                                break
                            if a + y in test_set:
                                ok = False
                                break
                        if ok and 2 * y in test_set:
                            ok = False
                        if ok:
                            # Accept the swap
                            current_set.discard(x)
                            current_set.add(y)
                            current = list(current_set)
                            best = len(current)
                            improved = True
                            break
                    if improved:
                        break
        return current

    def polish_pass(S, initial):
        """Final polish: try to add back any element that fits, in multiple passes."""
        current = list(initial)
        current_set = set(current)
        n = len(S)
        
        for _ in range(3):  # Multiple passes
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            added = False
            for y in S:
                if y in current_set:
                    continue
                ok = True
                for a in current_set:
                    if (y - a) in current_set:
                        ok = False
                        break
                    if a + y in current_set:
                        ok = False
                        break
                if ok and 2 * y in current_set:
                    ok = False
                if ok:
                    current.append(y)
                    current_set.add(y)
                    added = True
            if not added:
                break
        return current

    def solve_mwis(S, deadline):
        """
        Solves Max Independent Set on graph G where u~v iff u+v in S.
        deadline is absolute time.
        """
        n = len(S)
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if S[i] + S[j] in val_to_idx:
                    adj[i].add(j)
                    adj[j].add(i)
        
        def greedy_independent_set(adj, n):
            remaining = set(range(n))
            chosen = []
            while remaining:
                v = min(remaining, key=lambda x: len(adj[x] & remaining))
                chosen.append(v)
                remaining.discard(v)
                remaining -= adj[v]
            return chosen

        init_set = greedy_independent_set(adj, n)
        best_size = len(init_set)
        best_set = set(init_set)
        
        import sys
        sys.setrecursionlimit(10000)
        
        def bound(P):
            if not P:
                return 0
            colors = {}
            current_color = 0
            P_list = list(P)
            P_list.sort(key=lambda x: len(adj[x] & P), reverse=True)
            
            for v in P_list:
                used_colors = {colors[u] for u in adj[v] if u in colors}
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > current_color:
                    current_color = c
            
            return current_color + 1
        
        def bb(P, I):
            nonlocal best_size, best_set
            
            if time.time() > deadline:
                return
            
            if not P:
                if len(I) > best_size:
                    best_size = len(I)
                    best_set = set(I)
                return
            
            if len(I) + bound(P) <= best_size:
                return
            
            v = max(P, key=lambda x: len(adj[x] & P))
            
            I_new = I | {v}
            P_new = P - (adj[v] | {v})
            bb(P_new, I_new)
            
            P_new2 = P - {v}
            bb(P_new2, I)
            
        P_init = set(range(n))
        I_init = set()
        
        bb(P_init, I_init)
        
        return [S[i] for i in best_set]

    def solve_instance(S, time_budget):
        """Solve one instance using MIS B&B."""
        deadline = start_time + time_budget
        return solve_mwis(S, deadline)

    # Solve all three instances
    instances = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        instances.append(S)
    
    results = []
    for S in instances:
        if time.time() - start_time > DEADLINE - MARGIN:
            # Fallback to odd numbers
            results.append([x for x in S if x % 2 != 0])
        else:
            result = solve_instance(S, DEADLINE - MARGIN - (time.time() - start_time))
            # Verify
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
