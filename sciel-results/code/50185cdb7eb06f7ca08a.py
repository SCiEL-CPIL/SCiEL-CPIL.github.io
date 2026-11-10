import time
import random
from math import gcd

def construct():
    deadline = time.time() + 82  # safety margin under 90s kill
    N = 128

    # Precompute gcd matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute for each pair (a,b) the list of c that form a bad triple
    # bad_pair_c[a][b] = list of c (0-based) such that gcd(a+1,b+1)==gcd(a+1,c+1)==gcd(b+1,c+1)
    bad_pair_c = [[None]*N for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            ga = gcd_mat[a+1][b+1]
            lst = []
            for c in range(N):
                if c == a or c == b:
                    continue
                if gcd_mat[a+1][c+1] == ga and gcd_mat[b+1][c+1] == ga:
                    lst.append(c)
            bad_pair_c[a][b] = lst
            bad_pair_c[b][a] = lst

    # For each vertex v, list of pairs (a,b) that with v form a bad triple
    # conflict_pairs[v] = list of (a,b) with a<b, a!=v, b!=v
    conflict_pairs = [[] for _ in range(N)]
    for a in range(N):
        for b in range(a+1, N):
            for c in bad_pair_c[a][b]:
                conflict_pairs[c].append((a,b))
                conflict_pairs[a].append((b,c))
                conflict_pairs[b].append((a,c))

    # For fast membership: for each v, a set of (min,max) pairs
    conflict_set = [set() for _ in range(N)]
    for v in range(N):
        for (a,b) in conflict_pairs[v]:
            if a < b:
                conflict_set[v].add((a,b))
            else:
                conflict_set[v].add((b,a))

    # For each vertex v, count how many conflict pairs it has
    conflict_deg = [len(conflict_pairs[v]) for v in range(N)]

    def can_add(v, selected_set):
        """Check if v can be added to selected_set without creating a bad triple."""
        for (a, b) in conflict_pairs[v]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def greedy(order):
        """Greedy construction with given order."""
        selected = []
        selected_set = set()
        for v in order:
            if can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
        return selected, selected_set

    def local_improve(selected, selected_set):
        """Try to add any vertex that can fit."""
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if v in selected_set:
                    continue
                if can_add(v, selected_set):
                    selected.append(v)
                    selected_set.add(v)
                    improved = True
        return selected, selected_set

    def try_add_one(selected, selected_set):
        """Try to add one vertex that fits."""
        for v in range(N):
            if v in selected_set:
                continue
            if can_add(v, selected_set):
                selected.append(v)
                selected_set.add(v)
                return True
        return False

    def try_swap(selected, selected_set):
        """Try to remove one vertex and add one or more to improve size."""
        n = len(selected)
        for i in range(n):
            old = selected[i]
            # Remove old
            selected_set.discard(old)
            # Try to add multiple vertices
            added = 0
            added_list = []
            for v in range(N):
                if v in selected_set or v == old:
                    continue
                if can_add(v, selected_set):
                    selected_set.add(v)
                    added_list.append(v)
                    added += 1
            if added >= 1:
                # Net gain: added - 1 (we removed one)
                if added >= 1:
                    # Accept if we gained at least 0 (same size or better)
                    # Actually we want strictly better: added > 1
                    if added > 1:
                        selected = [x for x in selected if x != old] + added_list
                        return True
                    # If added == 1, same size, skip for now
            # Restore
            for v in added_list:
                selected_set.discard(v)
            selected_set.add(old)
        return False

    def try_swap2(selected, selected_set):
        """Try to remove two vertices and add three or more."""
        n = len(selected)
        for i in range(n):
            for j in range(i+1, n):
                old1, old2 = selected[i], selected[j]
                selected_set.discard(old1)
                selected_set.discard(old2)
                added_list = []
                for v in range(N):
                    if v in selected_set or v == old1 or v == old2:
                        continue
                    if can_add(v, selected_set):
                        selected_set.add(v)
                        added_list.append(v)
                if len(added_list) >= 3:
                    selected = [x for x in selected if x != old1 and x != old2] + added_list
                    return True
                for v in added_list:
                    selected_set.discard(v)
                selected_set.add(old1)
                selected_set.add(old2)
        return False

    # Main search
    best = []
    best_set = set()
    best_size = 0
    random.seed(12345)

    # Phase 1: Multiple random greedy + local improve
    for trial in range(2000):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 2: Structured orderings
    structured_orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda x: conflict_deg[x]),
        sorted(range(N), key=lambda x: -conflict_deg[x]),
        sorted(range(N), key=lambda x: (conflict_deg[x], x)),
    ]
    for order in structured_orders:
        if time.time() > deadline:
            break
        selected, selected_set = greedy(order)
        selected, selected_set = local_improve(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 3: Local search with swaps starting from best
    if best:
        selected = list(best)
        selected_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try single swaps
            if try_swap(selected, selected_set):
                improved = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best = list(selected)
                    best_set = set(selected)
            # Try 2-for-3 swaps
            if try_swap2(selected, selected_set):
                improved = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best = list(selected)
                    best_set = set(selected)

    # Phase 4: More randomized restarts with perturbation
    for trial in range(1500):
        if time.time() > deadline:
            break
        order = list(range(N))
        random.shuffle(order)
        selected, selected_set = greedy(order)
        # Perturb: remove a few random elements
        for _ in range(random.randint(1, 3)):
            if not selected:
                break
            idx = random.randint(0, len(selected) - 1)
            v = selected.pop(idx)
            selected_set.discard(v)
        selected, selected_set = local_improve(selected, selected_set)
        if len(selected) > best_size:
            best_size = len(selected)
            best = list(selected)
            best_set = set(selected)

    # Phase 5: Tabu-like search from best
    if best and time.time() < deadline - 5:
        selected = list(best)
        selected_set = set(best)
        tabu = {}  # vertex -> iteration when it became tabu
        tabu_tenure = 20
        it = 0
        no_improve = 0
        while time.time() < deadline - 3 and no_improve < 200:
            it += 1
            # Try all possible single moves: remove one, add one
            best_move = None
            best_gain = 0
            n = len(selected)
            for i in range(n):
                old = selected[i]
                if tabu.get(old, -1) > it - tabu_tenure and old in selected_set:
                    continue
                selected_set.discard(old)
                # Find best vertex to add
                best_add = None
                for v in range(N):
                    if v in selected_set or v == old:
                        continue
                    if can_add(v, selected_set):
                        best_add = v
                        break
                if best_add is not None:
                    selected_set.add(best_add)
                    gain = 0  # same size
                    # Check if this is an improvement in some sense
                    # For now, just accept if not tabu
                    if best_move is None or (old not in tabu or tabu[old] < it - tabu_tenure):
                        best_move = (old, best_add)
                        best_gain = 1
                selected_set.add(old)

            if best_move is not None:
                old, new = best_move
                selected_set.discard(old)
                selected_set.add(new)
                selected = [x for x in selected if x != old] + [new]
                tabu[new] = it
                if len(selected) > best_size:
                    best_size = len(selected)
                    best = list(selected)
                    best_set = set(selected)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # No valid move, try a random kick
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    v = selected.pop(idx)
                    selected_set.discard(v)
                    selected, selected_set = local_improve(selected, selected_set)
                    no_improve += 1

    # Final local improve on best
    if best:
        best_set = set(best)
        best, best_set = local_improve(list(best), best_set)
        if len(best) > best_size:
            best_size = len(best)

    if not best:
        best = list(range(1, N+1))  # fallback (may be invalid but nonempty)

    return sorted(best)
