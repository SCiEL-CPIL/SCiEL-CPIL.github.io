"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    deadline = time.time() + 82.0
    n = 128

    # Precompute gcd matrix
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]

    # For each element x, precompute the set of pairs (a,b) that form a bad triple with x
    # i.e., gcd(a,b) == gcd(a,x) == gcd(b,x)
    bad_pairs_for_x = {x: set() for x in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_mat[a][b]
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                    bad_pairs_for_x[x].add((a, b))

    # For fast validity checking, precompute for each x a set of all elements
    # that appear in any bad pair with x
    elements_in_bad_pairs = {x: set() for x in range(1, n + 1)}
    for x in range(1, n + 1):
        for (a, b) in bad_pairs_for_x[x]:
            elements_in_bad_pairs[x].add(a)
            elements_in_bad_pairs[x].add(b)

    def is_valid(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        bp = bad_pairs_for_x[x]
        if not bp:
            return True
        # Quick check: if no element from selected_set is in elements_in_bad_pairs[x], skip
        if not (selected_set & elements_in_bad_pairs[x]):
            return True
        sel = list(selected_set & elements_in_bad_pairs[x])
        for i in range(len(sel)):
            a = sel[i]
            for j in range(i + 1, len(sel)):
                b = sel[j]
                if a < b:
                    if (a, b) in bp:
                        return False
                else:
                    if (b, a) in bp:
                        return False
        return True

    # Greedy add with various orderings
    def greedy_add(order):
        selected = set()
        for x in order:
            if is_valid(x, selected):
                selected.add(x)
        return selected

    # Greedy remove: start with all, remove conflicts
    def greedy_remove():
        selected = set(range(1, n + 1))
        changed = True
        while changed:
            changed = False
            for x in list(selected):
                bp = bad_pairs_for_x[x]
                if not bp:
                    continue
                conflict = False
                sel_others = selected - {x}
                for (a, b) in bp:
                    if a in sel_others and b in sel_others:
                        conflict = True
                        break
                if conflict:
                    selected.remove(x)
                    changed = True
        return selected

    # Get initial from greedy remove
    best_set = greedy_remove()
    best_size = len(best_set)

    random.seed(42)

    # Multiple greedy restarts with different orderings
    for attempt in range(300):
        if time.time() > deadline - 15.0:
            break

        if attempt % 6 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 6 == 1:
            # Least constrained first
            order = sorted(range(1, n + 1), key=lambda x: len(bad_pairs_for_x[x]))
        elif attempt % 6 == 2:
            # Most constrained first
            order = sorted(range(1, n + 1), key=lambda x: -len(bad_pairs_for_x[x]))
        elif attempt % 6 == 3:
            # By number of prime factors (fewer = safer)
            def num_pf(x):
                count = 0
                d = x
                for p in range(2, int(x**0.5) + 1):
                    while d % p == 0:
                        count += 1
                        d //= p
                if d > 1:
                    count += 1
                return count
            order = sorted(range(1, n + 1), key=lambda x: num_pf(x))
        elif attempt % 6 == 4:
            # By largest prime factor descending
            def lpf(x):
                d = x
                for p in range(2, int(x**0.5) + 1):
                    if x % p == 0:
                        d = x // p
                        while d % p == 0:
                            d //= p
                        if d > 1:
                            break
                return d if d > 1 else x
            order = sorted(range(1, n + 1), key=lambda x: -lpf(x))
        else:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)

        candidate = greedy_add(order)
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)

    # Local search: try to add any element
    def try_add_any(current_set):
        for y in range(1, n + 1):
            if y not in current_set and is_valid(y, current_set):
                current_set.add(y)
                return True
        return False

    # 1-for-2 swap: remove x, try to add two new elements
    def local_search_1for2(current_set, max_rounds=100):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline - 12.0:
                break
            sel_list = list(current_set)
            random.shuffle(sel_list)
            for x in sel_list:
                if time.time() > deadline - 12.0:
                    return current_set
                remaining = current_set - {x}
                cands = [y for y in range(1, n + 1) if y not in remaining and y != x and is_valid(y, remaining)]
                if len(cands) < 2:
                    continue
                random.shuffle(cands)
                # Try pairs
                found = False
                for i in range(min(len(cands), 20)):
                    if found:
                        break
                    y1 = cands[i]
                    temp = remaining | {y1}
                    for j in range(i + 1, min(len(cands), 20)):
                        y2 = cands[j]
                        if is_valid(y2, temp):
                            current_set = temp | {y2}
                            improved = True
                            found = True
                            break
                if found:
                    break
        return current_set

    # 2-for-3 swap: remove two, add three
    def local_search_2for3(current_set, max_rounds=30):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline - 10.0:
                break
            sel_list = list(current_set)
            random.shuffle(sel_list)
            for i in range(min(len(sel_list), 10)):
                if time.time() > deadline - 10.0:
                    break
                x1 = sel_list[i]
                for j in range(i + 1, min(len(sel_list), 10)):
                    if time.time() > deadline - 10.0:
                        break
                    x2 = sel_list[j]
                    remaining = current_set - {x1, x2}
                    cands = [y for y in range(1, n + 1) if y not in remaining and y != x1 and y != x2 and is_valid(y, remaining)]
                    if len(cands) < 3:
                        continue
                    random.shuffle(cands)
                    found = False
                    for a in range(min(len(cands), 10)):
                        if found:
                            break
                        y1 = cands[a]
                        t1 = remaining | {y1}
                        for b in range(a + 1, min(len(cands), 10)):
                            if found:
                                break
                            y2 = cands[b]
                            if not is_valid(y2, t1):
                                continue
                            t2 = t1 | {y2}
                            for c in range(b + 1, min(len(cands), 10)):
                                if found:
                                    break
                                y3 = cands[c]
                                if is_valid(y3, t2):
                                    current_set = t2 | {y3}
                                    improved = True
                                    found = True
                                    break
                    if found:
                        break
                if improved:
                    break
        return current_set

    # Run local searches
    if time.time() < deadline - 25.0:
        best_set = local_search_1for2(best_set, max_rounds=200)
        best_size = len(best_set)

    # Try to add more
    while time.time() < deadline - 10.0:
        if not try_add_any(best_set):
            break
        best_size = len(best_set)

    # More 1-for-2 after additions
    if time.time() < deadline - 18.0:
        best_set = local_search_1for2(best_set, max_rounds=150)
        best_size = len(best_set)

    # 2-for-3
    if time.time() < deadline - 14.0:
        best_set = local_search_2for3(best_set, max_rounds=20)
        best_size = len(best_set)

    # Final add pass
    while time.time() < deadline - 6.0:
        if not try_add_any(best_set):
            break
        best_size = len(best_set)

    # Stochastic local search: random remove/reinsert
    if time.time() < deadline - 12.0:
        for _ in range(300):
            if time.time() > deadline - 10.0:
                break
            if not best_set:
                break
            k = random.choice([1, 1, 2])
            to_remove = random.sample(list(best_set), min(k, len(best_set)))
            temp = best_set - set(to_remove)
            # Try to add random elements
            for _ in range(8):
                y = random.randint(1, n)
                if y not in temp and is_valid(y, temp):
                    temp.add(y)
            # Try to add any
            for y in range(1, n + 1):
                if y not in temp and is_valid(y, temp):
                    temp.add(y)
            if len(temp) > len(best_set):
                best_set = temp
                best_size = len(best_set)

    # Final polish
    while time.time() < deadline - 3.0:
        if not try_add_any(best_set):
            break

    return list(best_set)
# EVOLVE-BLOCK-END
