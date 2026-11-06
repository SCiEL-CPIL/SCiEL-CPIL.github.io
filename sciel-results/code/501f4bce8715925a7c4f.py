import math
import time
import random

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill
    random.seed(42)

    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)

    # Conflicts restricted to the candidate set.
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    def time_up(budget=None):
        if budget is None:
            return time.time() - start > deadline - 3.0
        return time.time() - start > budget

    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def greedy_min_degree(candidates_list):
        remaining = set(candidates_list)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflict[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            independent_set.add(chosen)
            remaining.remove(chosen)
            remaining -= conflict[chosen]
        return independent_set

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 2 nodes (size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

            # 3. Try 2-swap: remove 2 nodes, add 3+ nodes for net gain
            if not improved_flag and not time_up(time_budget):
                s_list = list(s)
                for i in range(len(s_list)):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, len(s_list)):
                        if time_up(time_budget):
                            break
                        u1, u2 = s_list[i], s_list[j]
                        s.remove(u1)
                        s.remove(u2)
                        
                        addable = []
                        for v in candidates:
                            if v not in s and v != u1 and v != u2:
                                if try_add_fast(s, v):
                                    addable.append(v)
                        
                        if len(addable) >= 3:
                            for v in addable:
                                s.add(v)
                            improved_flag = True
                            break
                        else:
                            s.add(u1)
                            s.add(u2)
                    if improved_flag:
                        break

        return s

    # 1. Start with greedy min-degree solution
    best_solution = greedy_min_degree(candidates)
    best_size = len(best_solution)

    # 2. Local search on the greedy solution
    best_solution = local_search_improve(best_solution, deadline - 20.0)
    best_size = len(best_solution)

    # 3. Multiple random restarts with different orderings
    for restart in range(300):
        if time_up(deadline - 10.0):
            break
        
        # Randomized greedy with different strategies
        strategy = restart % 3
        if strategy == 0:
            # Pure random ordering
            shuffled = candidates[:]
            random.shuffle(shuffled)
            current_set = set()
            for v in shuffled:
                if try_add_fast(current_set, v):
                    current_set.add(v)
        elif strategy == 1:
            # Temperature-based random selection
            rem = set(candidates)
            current_set = set()
            while rem:
                degs = {}
                for node in rem:
                    degs[node] = len(conflict[node] & rem)
                min_d = min(degs.values()) if degs else 0
                tau = max(1.0, min_d * 0.5)
                weights = []
                total_w = 0.0
                for node in rem:
                    w = math.exp(-degs[node] / tau)
                    weights.append(w)
                    total_w += w
                r = random.random() * total_w
                cum = 0.0
                chosen = list(rem)[0]
                for node, w in zip(rem, weights):
                    cum += w
                    if r <= cum:
                        chosen = node
                        break
                current_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
        else:
            # Min-degree with random tie-breaking
            rem = set(candidates)
            current_set = set()
            while rem:
                min_deg = float('inf')
                candidates_min = []
                for v in rem:
                    deg = len(conflict[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        candidates_min = [v]
                    elif deg == min_deg:
                        candidates_min.append(v)
                chosen = random.choice(candidates_min)
                current_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
        
        # Local search on this result
        ls_result = local_search_improve(current_set, deadline - 10.0)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_solution = ls_result

    # 4. Final polish: try one more round of local search with tighter budget
    if not time_up(deadline - 2.0):
        best_solution = local_search_improve(best_solution, deadline - 1.0)

    # 5. Safety verification: ensure no pair (a<=b) has a square sum.
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
        # fallback: deterministic min-degree greedy (always valid)
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
