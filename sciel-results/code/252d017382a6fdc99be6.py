"""Rewritten solver for large sets avoiding reciprocal Schur triples."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 180

    # Precompute all forbidden triples: (a, b, c) with a = bc/(b+c), a,b,c distinct in [1,180]
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triples.append((a, b, c))

    # Build conflict structure: for each element x, list of pairs (y,z) such that {x,y,z} is a forbidden triple
    from collections import defaultdict
    conflict_pairs = defaultdict(set)
    for (a, b, c) in triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))

    def can_add(S, x):
        """Check if x can be added to set S without creating a forbidden triple."""
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True

    # Multiple greedy strategies
    def greedy(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    # Try several orderings
    best_set = set()
    
    # Order 1: increasing
    S = greedy(list(range(1, N + 1)))
    if len(S) > len(best_set):
        best_set = S

    # Order 2: decreasing
    S = greedy(list(range(N, 0, -1)))
    if len(S) > len(best_set):
        best_set = S

    # Order 3: by conflict degree ascending
    degrees = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    order_asc = sorted(range(1, N + 1), key=lambda x: degrees[x])
    S = greedy(order_asc)
    if len(S) > len(best_set):
        best_set = S

    # Order 4: by conflict degree descending
    order_desc = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    S = greedy(order_desc)
    if len(S) > len(best_set):
        best_set = S

    # Randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Random perturbation of the order
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Mix in some structure: start with low-conflict elements
        order = sorted(order, key=lambda x: degrees[x] * 0.5 + random.random())
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S

    # Local search: try to add any element that fits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            if can_add(best_set, x):
                best_set.add(x)
                improved = True
                break

    # Simulated annealing to escape local optima
    if time.time() < deadline - 5:
        current = set(best_set)
        current_size = len(current)
        best_sa = set(best_set)
        best_size = len(best_sa)
        
        T = 5.0  # initial temperature
        T_min = 0.01
        cooling = 0.995
        
        while time.time() < deadline - 2:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element not in current
                candidates = [x for x in range(1, N + 1) if x not in current]
                if not candidates:
                    break
                x = random.choice(candidates)
                if can_add(current, x):
                    new_set = set(current)
                    new_set.add(x)
                    delta = 1
                    if delta > 0 or random.random() < __import__('math').exp(-delta / T):
                        current = new_set
                        if len(current) > best_size:
                            best_sa = set(current)
                            best_size = len(current)
                else:
                    # Can't add directly; try removing one conflicting element and adding
                    # Find a conflict
                    for (y, z) in conflict_pairs[x]:
                        if y in current and z in current:
                            # Try removing y
                            if random.random() < 0.5:
                                removed = y
                            else:
                                removed = z
                            new_set = set(current)
                            new_set.discard(removed)
                            if can_add(new_set, x):
                                new_set.add(x)
                                delta = len(new_set) - len(current)
                                if delta >= 0 or random.random() < __import__('math').exp(-delta / T):
                                    current = new_set
                                    if len(current) > best_size:
                                        best_sa = set(current)
                                        best_size = len(current)
                            break
            elif move_type < 0.7:
                # Try to remove a random element from current
                if not current:
                    break
                x = random.choice(list(current))
                new_set = set(current)
                new_set.discard(x)
                delta = -1
                if random.random() < __import__('math').exp(-delta / T):
                    current = new_set
            else:
                # Swap: remove one, try to add another
                if not current:
                    break
                x = random.choice(list(current))
                candidates = [y for y in range(1, N + 1) if y not in current]
                if not candidates:
                    break
                y = random.choice(candidates)
                new_set = set(current)
                new_set.discard(x)
                if can_add(new_set, y):
                    new_set.add(y)
                    delta = 0  # same size
                    if random.random() < 0.3:  # accept some swaps to explore
                        current = new_set
                        if len(current) > best_size:
                            best_sa = set(current)
                            best_size = len(current)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if len(best_sa) > len(best_set):
            best_set = best_sa

    # Final polish: add any remaining elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            if can_add(best_set, x):
                best_set.add(x)
                improved = True
                break

    # Try 2-for-1 swaps: remove one, try to add two
    if time.time() < deadline - 5:
        for x in list(best_set):
            if time.time() > deadline:
                break
            saved = set(best_set)
            best_set.discard(x)
            added = []
            for y in range(1, N + 1):
                if y in best_set:
                    continue
                if time.time() > deadline:
                    break
                if can_add(best_set, y):
                    best_set.add(y)
                    added.append(y)
            if len(best_set) > len(saved):
                pass  # Keep the improvement
            else:
                best_set = set(saved)

    # Final final polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if time.time() > deadline:
                break
            if can_add(best_set, x):
                best_set.add(x)
                improved = True
                break

    return sorted(best_set)
# EVOLVE-BLOCK-END
