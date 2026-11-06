"""Rewritten program with stronger algorithm for integer set pair-sum non-divisibility."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = t0 + 75.0  # safety margin before 90s kill

    N = 256
    all_nums = list(range(1, N + 1))
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]

    # Fast conflict check using algebraic identity:
    # (a+b) | ab  iff  (a+b) | a^2  (since ab = a(a+b) - a^2, so ab mod (a+b) = -a^2 mod (a+b))
    # So (a+b) | ab  iff  a^2 % (a+b) == 0
    def conflicts(a, b):
        return a * a % (a + b) == 0

    # Precompute full conflict graph using the faster check
    conflict_graph = {}  # x -> set of y where x!=y and (x*y)%(x+y)==0
    for x in all_nums:
        cs = set()
        for y in all_nums:
            if x == y:
                continue
            if conflicts(x, y):
                cs.add(y)
        conflict_graph[x] = cs

    # Precompute degree
    degrees = {x: len(conflict_graph[x]) for x in all_nums}
    
    # Greedy independent set with multiple orderings
    def greedy_independent_set(order):
        selected = set()
        for x in order:
            if not (conflict_graph[x] & selected):
                selected.add(x)
        return selected

    # Enhanced local search with better swap strategies
    def local_search(current_set, max_iter=500):
        if time.time() > DEADLINE:
            return current_set
        cur = set(current_set)
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < DEADLINE:
            improved = False
            iterations += 1
            
            # Try adding each unselected node
            for x in all_nums:
                if x in cur:
                    continue
                if not (conflict_graph[x] & cur):
                    cur.add(x)
                    improved = True
                    break
            
            # Try 1-for-1 swap: remove one, add another (net size +1)
            if not improved:
                cur_list = list(cur)
                for rem in cur_list:
                    trial = cur - {rem}
                    for x in all_nums:
                        if x in trial:
                            continue
                        if not (conflict_graph[x] & trial):
                            trial.add(x)
                            if len(trial) > len(cur):
                                cur = trial
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-for-k swaps: remove 2, add k, net gain if k > 2
            if not improved and time.time() < DEADLINE:
                cur_list = list(cur)
                n_cur = len(cur_list)
                # Limit pair search to keep it fast
                limit = min(n_cur, 50)
                for i in range(limit):
                    if time.time() > DEADLINE:
                        break
                    for j in range(i+1, limit):
                        rem1, rem2 = cur_list[i], cur_list[j]
                        trial = cur - {rem1, rem2}
                        added = 0
                        for x in all_nums:
                            if x in trial:
                                continue
                            if not (conflict_graph[x] & trial):
                                trial.add(x)
                                added += 1
                        if added > 2:
                            cur = trial
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 1-for-1 swap that keeps size same but may enable future adds
            # (skip for now to save time)
        
        return cur

    # Multiple greedy orderings
    best_set = set()
    best_size = 0

    # Ordering 1: by degree ascending (prefer low-conflict nodes first)
    order1 = sorted(all_nums, key=lambda x: (degrees[x], x))
    s1 = greedy_independent_set(order1)
    s1 = local_search(s1, max_iter=300)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1

    # Ordering 2: by value ascending
    order2 = sorted(all_nums, key=lambda x: x)
    s2 = greedy_independent_set(order2)
    s2 = local_search(s2, max_iter=300)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # Ordering 3: by value descending
    order3 = sorted(all_nums, key=lambda x: -x)
    s3 = greedy_independent_set(order3)
    s3 = local_search(s3, max_iter=300)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3

    # Ordering 4: by "conflict density" - prefer nodes whose conflicts are also high-degree
    # (removing high-degree conflicts is more valuable)
    order4 = sorted(all_nums, key=lambda x: (degrees[x], -x))
    s4 = greedy_independent_set(order4)
    s4 = local_search(s4, max_iter=300)
    if len(s4) > best_size:
        best_size = len(s4)
        best_set = s4

    # Ordering 5: random restarts with more iterations
    rng = random.Random(42)
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > DEADLINE:
            break
        random_order = all_nums[:]
        rng.shuffle(random_order)
        sr = greedy_independent_set(random_order)
        sr = local_search(sr, max_iter=150)
        if len(sr) > best_size:
            best_size = len(sr)
            best_set = sr

    # More aggressive local search on the best
    if time.time() < DEADLINE:
        best_set = local_search(best_set, max_iter=2000)

    # Final verification
    result = sorted(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: find a valid subset by removing conflicting elements
        result = [1]
        for x in all_nums:
            if x == 1:
                continue
            if not (conflict_graph[x] & set(result)):
                result.append(x)
        result = sorted(result)

    return result
# EVOLVE-BLOCK-END
