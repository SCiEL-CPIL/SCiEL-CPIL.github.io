"""Rewritten program: maximize subset of [1,128] with no equal-lcm triples."""
import time
from math import gcd

# Deadline guard: kill before 90s wall-clock; use 80s budget for safety
_DEADLINE = 80.0

def construct():
    import time
    start_time = time.time()
    
    # Precompute LCM for all pairs in [1, 128]
    # lcm(a,b) = a*b // gcd(a,b)
    # We need to find subsets with no three distinct a,b,c such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    # Key insight: When do three numbers have all pairwise LCMs equal?
    # Let L = lcm(a,b) = lcm(a,c) = lcm(b,c)
    # This means a,b,c all divide L, and for each pair, their lcm is L.
    # Equivalently, a,b,c are divisors of L such that no two of them have
    # lcm < L. In other words, for every pair (x,y) in {a,b,c}, lcm(x,y) = L.
    
    # Strategy: Greedy with multiple orderings, plus local search / simulated annealing.
    
    # Precompute gcd and lcm tables
    n_max = 128
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    lcm_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i, n_max + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            lcm_table[i][j] = i * j // g
            lcm_table[j][i] = lcm_table[i][j]
    
    # Check if adding x to a set S creates an equal-lcm triple
    # A triple (a, b, x) is bad if lcm(a,b) == lcm(a,x) == lcm(b,x)
    
    def is_bad_triple(a, b, x):
        return lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]
    
    # Greedy: try multiple orderings and keep the best
    def greedy_with_order(order):
        selected = []
        selected_set = set()
        for x in order:
            if x in selected_set:
                continue
            # Check all pairs in selected
            ok = True
            for i in range(len(selected)):
                a = selected[i]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    if is_bad_triple(a, b, x):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
                selected_set.add(x)
            if time.time() - start_time > _DEADLINE - 5:
                break
        return selected
    
    # Try natural order and reverse order
    candidates = []
    candidates.append(greedy_with_order(range(1, n_max + 1)))
    candidates.append(greedy_with_order(range(n_max, 0, -1)))
    
    # Try random orders
    import random
    rng = random.Random(42)
    for trial in range(50):
        if time.time() - start_time > _DEADLINE - 10:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        cand = greedy_with_order(order)
        candidates.append(cand)
    
    # Local search: start from best greedy, try swaps and additions
    best = max(candidates, key=len)
    best_set = set(best)
    
    def count_bad_triples(s):
        """Count number of equal-lcm triples in set s."""
        s_list = sorted(s)
        count = 0
        n = len(s_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        count += 1
        return count
    
    # Verify best is valid
    assert count_bad_triples(best_set) == 0, "Greedy result should be valid"
    
    # Local search: try to add elements, remove elements to allow more additions
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > _DEADLINE - 15:
            break
        # Try adding each missing element
        for x in range(1, n_max + 1):
            if x in best_set:
                continue
            new_set = best_set | {x}
            if count_bad_triples(new_set) == 0:
                best_set = new_set
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add one (or two)
        if time.time() - start_time > _DEADLINE - 15:
            break
        best_list = sorted(best_set)
        for i in range(len(best_list)):
            for x in range(1, n_max + 1):
                if x in best_set:
                    continue
                new_set = best_set - {best_list[i]} | {x}
                if len(new_set) == len(best_set):  # same size, but maybe allows future adds
                    # Only accept if it doesn't create triples AND we can then add more
                    if count_bad_triples(new_set) == 0:
                        # Check if we can add any element to new_set
                        for y in range(1, n_max + 1):
                            if y in new_set:
                                continue
                            test_set = new_set | {y}
                            if count_bad_triples(test_set) == 0:
                                best_set = test_set
                                improved = True
                                break
                        if improved:
                            break
            if improved:
                break
    
    # Simulated annealing / random restart local search
    if time.time() - start_time < _DEADLINE - 20:
        best_size = len(best_set)
        for sa_iter in range(200):
            if time.time() - start_time > _DEADLINE - 10:
                break
            # Start from current best, perturb
            cur_set = set(best_set)
            # Remove 1-3 random elements
            to_remove = random.sample(sorted(cur_set), min(3, len(cur_set)))
            for t in to_remove:
                cur_set.discard(t)
            # Try to greedily fill back
            order = list(range(1, n_max + 1))
            random.shuffle(order)
            for x in order:
                if x in cur_set:
                    continue
                new_set = cur_set | {x}
                if count_bad_triples(new_set) == 0:
                    cur_set = new_set
            if len(cur_set) > best_size:
                best_set = cur_set
                best_size = len(cur_set)
    
    # Final verification
    final_list = sorted(best_set)
    # Double-check validity
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            for k in range(j + 1, len(final_list)):
                a, b, c = final_list[i], final_list[j], final_list[k]
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    # Should not happen; remove c to fix
                    final_list = [x for x in final_list if x != c]
                    break
    
    if not final_list:
        return [1]
    return final_list
