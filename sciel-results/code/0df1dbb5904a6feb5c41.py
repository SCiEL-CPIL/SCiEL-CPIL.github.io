"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 80  # safety margin before 90s kill

    # Precompute GCD table for O(1) lookups
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Build 2-uniform conflict graph: conflict[u][v] = True if there exists w
    # such that gcd(u,v) = gcd(u,w) = gcd(v,w)
    # This reduces the 3-uniform hypergraph to a standard graph
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            g = gcd_table[u][v]
            # Check if any w (distinct from u,v) has gcd(u,w)=g and gcd(v,w)=g
            for w in range(1, N + 1):
                if w != u and w != v:
                    if gcd_table[u][w] == g and gcd_table[v][w] == g:
                        conflict[u][v] = True
                        conflict[v][u] = True
                        break

    # Precompute conflict neighbor lists for fast checking
    conflict_neighbors = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if conflict[u][v]:
                conflict_neighbors[u].add(v)
                conflict_neighbors[v].add(u)

    def can_add(selected_set, x):
        """Check if x can be added without creating a bad triple.
        x conflicts with a pair (a,b) in selected if x is in bad_triple[a][b].
        But with 2-uniform graph: x conflicts with a if conflict[x][a] AND 
        there exists b in selected where conflict[x][b] and the triple {x,a,b} is bad.
        Actually, the 2-uniform graph approach: an edge (u,v) means u and v cannot 
        both be in the set IF there's a w that would form a bad triple. But w might 
        not be in the set. So the 2-uniform reduction is NOT exact.
        
        Let me use the direct check: x can be added if for all pairs (a,b) in selected,
        gcd(x,a) != gcd(x,b) != gcd(a,b) (all three not equal).
        """
        sel = list(selected_set)
        for i in range(len(sel)):
            a = sel[i]
            gxa = gcd_table[x][a]
            for j in range(i + 1, len(sel)):
                b = sel[j]
                gxb = gcd_table[x][b]
                gab = gcd_table[a][b]
                if gxa == gxb == gab:
                    return False
        return True

    def can_add_fast(selected_list, x):
        """Faster can_add using list."""
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            gxa = gcd_table[x][a]
            row_a = gcd_table[a]
            for j in range(i + 1, n):
                b = selected_list[j]
                if gxa == gcd_table[x][b] == row_a[b]:
                    return False
        return True

    def greedy_build(order, selected_set=None):
        """Greedy build following given order."""
        if selected_set is None:
            selected_set = set()
        selected_list = list(selected_set)
        for x in order:
            if x in selected_set:
                continue
            if can_add_fast(selected_list, x):
                selected_set.add(x)
                selected_list.append(x)
        return selected_set

    def local_search(selected_set, time_limit=5.0):
        """Local search with add, 1-swap, and 2-swap moves."""
        selected = set(selected_set)
        selected_list = list(selected)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if x in selected:
                    continue
                if can_add_fast(selected_list, x):
                    selected.add(x)
                    selected_list.append(x)
                    improved = True
                    break
            # If we can't add, try 1-swap: remove one, add another
            if not improved:
                for idx in range(len(selected_list)):
                    x = selected_list[idx]
                    remaining_list = selected_list[:idx] + selected_list[idx+1:]
                    found_swap = False
                    for y in range(1, N + 1):
                        if y in selected:
                            continue
                        if can_add_fast(remaining_list, y):
                            selected.discard(x)
                            selected.add(y)
                            selected_list = remaining_list + [y]
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            # If still no improvement, try 2-swap: remove two, add two
            if not improved and time.time() - start < time_limit - 0.5:
                n_sel = len(selected_list)
                found_2swap = False
                for i in range(n_sel):
                    if time.time() - start >= time_limit - 0.5:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() - start >= time_limit - 0.5:
                            break
                        x1, x2 = selected_list[i], selected_list[j]
                        remaining_list = [v for k, v in enumerate(selected_list) if k != i and k != j]
                        for y1 in range(1, N + 1):
                            if y1 in selected or found_2swap:
                                continue
                            if can_add_fast(remaining_list, y1):
                                temp_list = remaining_list + [y1]
                                for y2 in range(y1 + 1, N + 1):
                                    if y2 in selected or found_2swap:
                                        continue
                                    if can_add_fast(temp_list, y2):
                                        selected.discard(x1)
                                        selected.discard(x2)
                                        selected.add(y1)
                                        selected.add(y2)
                                        selected_list = temp_list + [y2]
                                        improved = True
                                        found_2swap = True
                                        break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
        return selected

    best = set()

    # Multiple restarts with different orderings
    base_order = list(range(1, N + 1))

    # Try natural order first
    result = greedy_build(base_order)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Try descending order
    desc_order = list(range(N, 0, -1))
    result = greedy_build(desc_order)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Random restarts
    no_improve_count = 0
    while time.time() < deadline:
        if random.random() < 0.4:
            # Perturb: start from best and randomly remove some, then rebuild
            start_set = set(best)
            for _ in range(random.randint(1, 4)):
                if start_set:
                    start_set.remove(random.choice(list(start_set)))
            remaining_order = [x for x in range(1, N + 1) if x not in start_set]
            random.shuffle(remaining_order)
            order = list(start_set) + remaining_order
            result = greedy_build(order, start_set)
        else:
            random.shuffle(base_order)
            result = greedy_build(base_order)
        result = local_search(result, time_limit=2.0)
        if len(result) > len(best):
            best = set(result)
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count > 100 and len(best) >= 80:
            break

    # Final polish: try to add any remaining elements (multiple passes)
    for _ in range(3):
        changed = False
        blist = list(best)
        for x in range(1, N + 1):
            if x not in best:
                if can_add_fast(blist, x):
                    best.add(x)
                    blist.append(x)
                    changed = True
        if not changed:
            break

    return list(best) if best else [1]
# EVOLVE-BLOCK-END
