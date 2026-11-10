"""Simulated annealing on conflict graph for reciprocal Schur triple avoidance."""
import random
import time
import math

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))

    # Build conflict_pairs[x] = list of (y, z) such that {x, y, z} is a forbidden triple
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for a, b, c in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Build neighbor sets for degree-based ordering
    neighbors = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (y, z) in conflict_pairs[x]:
            if y != x and z != x:
                neighbors[x].add(y)
                neighbors[x].add(z)
    degree = {x: len(neighbors[x]) for x in range(1, N + 1)}

    def count_conflicts(x, S):
        """Count how many forbidden triples would be formed if x were in S."""
        count = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                count += 1
        return count

    def total_conflicts(S):
        """Total number of forbidden triples in S."""
        count = 0
        for a in S:
            for (b, c) in conflict_pairs[a]:
                if b in S and c in S:
                    count += 1
        return count

    def has_triple(S):
        for a in S:
            for (b, c) in conflict_pairs[a]:
                if b in S and c in S:
                    return True
        return False

    def greedy_order(order):
        S = set()
        for x in order:
            if count_conflicts(x, S) == 0:
                S.add(x)
        return S

    # Phase 1: Get a good initial solution via degree-based greedy
    degree_order = sorted(range(1, N + 1), key=lambda x: degree[x])
    best = greedy_order(degree_order)
    best_size = len(best)

    # Try a few other orderings to get the best starting point
    for order in [list(reversed(degree_order)), list(range(N, 0, -1)), list(range(1, N + 1))]:
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Random restarts for initial solution
    for _ in range(100):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Phase 2: Simulated annealing
    # State: set S. Energy: -|S| + penalty * total_conflicts(S)
    # We want to maximize |S| subject to total_conflicts = 0.
    # Use energy = -|S| + 10 * total_conflicts(S) (minimize energy)
    
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.99995  # Slow cooling
    penalty = 10.0  # Penalty per conflict
    
    current_S = set(best)
    current_energy = -len(current_S) + penalty * total_conflicts(current_S)
    best_S = set(current_S)
    best_energy = current_energy
    best_size = len(current_S)
    
    T = T_start
    step = 0
    no_improve_count = 0
    
    while T > T_end and time.time() < deadline:
        step += 1
        
        # Choose a random move: 1-flip or 2-flip
        if rng.random() < 0.7:
            # 1-flip: add or remove a random element
            if rng.random() < 0.6:
                # Try adding an element not in S
                candidates = [x for x in range(1, N + 1) if x not in current_S]
                if not candidates:
                    T *= cooling_rate
                    continue
                x = rng.choice(candidates)
                new_S = current_S | {x}
            else:
                # Try removing an element from S
                if not current_S:
                    T *= cooling_rate
                    continue
                x = rng.choice(list(current_S))
                new_S = current_S - {x}
        else:
            # 2-flip: swap two elements
            in_S = list(current_S)
            not_in_S = [x for x in range(1, N + 1) if x not in current_S]
            if len(in_S) < 1 or len(not_in_S) < 1:
                T *= cooling_rate
                continue
            y = rng.choice(in_S)
            x = rng.choice(not_in_S)
            new_S = (current_S - {y}) | {x}
        
        # Compute energy difference
        new_conflicts = total_conflicts(new_S)
        new_energy = -len(new_S) + penalty * new_conflicts
        delta_E = new_energy - current_energy
        
        # Accept or reject
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_S = new_S
            current_energy = new_energy
            
            # Update best
            if total_conflicts(current_S) == 0 and len(current_S) > best_size:
                best_S = set(current_S)
                best_size = len(current_S)
                best_energy = current_energy
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # Every 1000 steps, try to polish if we have a valid solution
        if step % 500 == 0 and total_conflicts(current_S) == 0:
            # Try to add any element that doesn't create conflicts
            for x in range(1, N + 1):
                if x not in current_S and count_conflicts(x, current_S) == 0:
                    current_S.add(x)
                    current_energy = -len(current_S) + penalty * total_conflicts(current_S)
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_size = len(current_S)
        
        # Early termination if stuck
        if no_improve_count > 50000:
            break
    
    # Phase 3: Final polish on the best solution found
    final_S = set(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            if count_conflicts(x, final_S) == 0:
                final_S.add(x)
                improved = True
                break
    
    # Phase 4: Try 1-for-1 swaps on the polished solution
    for _ in range(5000):
        if time.time() >= deadline:
            break
        swapped = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            c_in_S = count_conflicts(x, final_S)
            if c_in_S == 0:
                continue
            for y in list(final_S):
                resolved = True
                for (u, v) in conflict_pairs[x]:
                    if u in final_S and v in final_S:
                        if u != y and v != y:
                            resolved = False
                            break
                if resolved:
                    final_S.discard(y)
                    final_S.add(x)
                    swapped = True
                    break
            if swapped:
                break
        if not swapped:
            break
    
    # Final polish after swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in final_S:
                continue
            if count_conflicts(x, final_S) == 0:
                final_S.add(x)
                improved = True
                break
    
    # Final verification
    if has_triple(final_S):
        final_S = set(range(91, N + 1))
    
    return list(final_S)
