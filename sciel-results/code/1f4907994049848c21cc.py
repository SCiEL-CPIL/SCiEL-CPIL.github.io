import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_used_pairs(triples):
        used = set()
        for a, b, c in triples:
            used.add((a, b) if a < b else (b, a))
            used.add((b, c) if b < c else (c, b))
            used.add((a, c) if a < c else (c, a))
        return used

    def calculate_alpha(triples):
        """Compute the independence number of the hypergraph defined by triples."""
        if not triples:
            return 16

        triple_masks = []
        for a, b, c in triples:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)

        max_size = 0

        def backtrack(mask, size, start_idx):
            nonlocal max_size
            if size > max_size:
                max_size = size

            # Pruning
            if size + (16 - start_idx) <= max_size:
                return

            for i in range(start_idx, 16):
                new_mask = mask | (1 << i)
                is_independent = True
                for tm in triple_masks:
                    if (new_mask & tm) == tm:
                        is_independent = False
                        break

                if is_independent:
                    backtrack(new_mask, size + 1, i + 1)

        backtrack(0, 0, 0)
        return max_size

    def is_linear(triples):
        used_pairs = set()
        for a, b, c in triples:
            pairs = [(a, b), (b, c), (a, c)]
            for p in pairs:
                p_sorted = tuple(sorted(p))
                if p_sorted in used_pairs:
                    return False
                used_pairs.add(p_sorted)
        return True

    def greedy_packing(seed):
        random.seed(seed)
        points = list(range(16))
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)

        triples = []
        used_pairs = set()

        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (b, c) if b < c else (c, b)
            p3 = (a, c) if a < c else (c, a)

            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                triples.append([a, b, c])
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)

        return triples

    def local_search(system, max_steps, time_limit):
        """
        Improve the system by swapping points in triples to reduce independence number.
        """
        current_system = [list(t) for t in system]
        current_score = calculate_alpha(current_system)

        def get_used_pairs(sys):
            used = set()
            for a, b, c in sys:
                used.add((a, b) if a < b else (b, a))
                used.add((b, c) if b < c else (c, b))
                used.add((a, c) if a < c else (c, a))
            return used

        used_pairs = get_used_pairs(current_system)

        step = 0
        while step < max_steps and time.time() < time_limit:
            step += 1
            if time.time() > time_limit:
                break

            improved = False
            idx = random.randint(0, len(current_system) - 1)
            t = current_system[idx]

            for k in range(3):
                old_point = t[k]
                candidates = [p for p in range(16) if p not in t]
                random.shuffle(candidates)

                for new_point in candidates:
                    new_t = t.copy()
                    new_t[k] = new_point

                    a, b, c = t
                    old_pairs = [(a, b) if a < b else (b, a),
                                 (b, c) if b < c else (c, b),
                                 (a, c) if a < c else (c, a)]

                    a, b, c = new_t
                    new_pairs = [(a, b) if a < b else (b, a),
                                 (b, c) if b < c else (c, b),
                                 (a, c) if a < c else (c, a)]

                    for p in old_pairs:
                        used_pairs.remove(p)

                    valid = True
                    for p in new_pairs:
                        if p in used_pairs:
                            valid = False
                            break

                    if valid:
                        for p in new_pairs:
                            used_pairs.add(p)

                        current_system[idx] = new_t

                        new_score = calculate_alpha(current_system)

                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            for p in new_pairs:
                                used_pairs.remove(p)
                            for p in old_pairs:
                                used_pairs.add(p)
                            current_system[idx] = t
                    else:
                        for p in old_pairs:
                            used_pairs.add(p)

                    if improved:
                        break
                if improved:
                    break
            if not improved and step % 100 == 0:
                if step > 5000:
                    break

        return current_system, current_score

    def structural_search(time_limit):
        """
        New structural approach: Start from a random maximal packing, then use
        a more aggressive move set that includes triple removals and re-additions
        to escape local optima.
        """
        system = greedy_packing(random.randint(0, 10000))
        current_score = calculate_alpha(system)
        used_pairs = get_used_pairs(system)

        def remove_triple(idx):
            nonlocal system, used_pairs
            t = system[idx]
            a, b, c = t
            pairs = [(a, b) if a < b else (b, a),
                     (b, c) if b < c else (c, b),
                     (a, c) if a < c else (c, a)]
            for p in pairs:
                used_pairs.remove(p)
            system.pop(idx)

        def add_triple(triple):
            nonlocal system, used_pairs
            a, b, c = triple
            pairs = [(a, b) if a < b else (b, a),
                     (b, c) if b < c else (c, b),
                     (a, c) if a < c else (c, a)]
            system.append(triple)
            for p in pairs:
                used_pairs.add(p)

        # Try to improve by removing triples that are "redundant" for independence
        # and re-adding different triples
        for _ in range(1000):
            if time.time() > time_limit:
                break

            # Remove a random triple
            if not system:
                break
            idx = random.randint(0, len(system) - 1)
            old_triple = system[idx]
            remove_triple(idx)

            new_score = calculate_alpha(system)
            if new_score < current_score:
                current_score = new_score
                continue
            else:
                add_triple(old_triple)
                break

        # Now try to add new triples that don't conflict
        all_triples = list(combinations(range(16), 3))
        random.shuffle(all_triples)
        for t in all_triples:
            if time.time() > time_limit:
                break
            a, b, c = t
            pairs = [(a, b) if a < b else (b, a),
                     (b, c) if b < c else (c, b),
                     (a, c) if a < c else (c, a)]
            if all(p not in used_pairs for p in pairs):
                add_triple(t)
                new_score = calculate_alpha(system)
                if new_score < current_score:
                    current_score = new_score
                else:
                    # Revert
                    remove_triple(len(system) - 1)

        return system, current_score

    best_system = None
    best_score = 16

    # Run multiple instances with different strategies
    for i in range(50):
        if time.time() > deadline - 20.0:
            break

        if i % 2 == 0:
            # Greedy + Local Search
            init_system = greedy_packing(i)
            search_start = time.time()
            optimized_system, score = local_search(init_system, max_steps=2000, time_limit=search_start + 0.1)
        else:
            # Structural Search
            search_start = time.time()
            optimized_system, score = structural_search(search_start + 0.1)

        if score < best_score:
            best_score = score
            best_system = optimized_system

            if best_score <= 5:
                break

    if best_system is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_system
