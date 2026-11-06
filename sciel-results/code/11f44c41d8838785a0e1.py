"""Hypergraph Ramsey R(4,5;3) coloring: avoid red K4^3 and blue K5^3."""
import time
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - elapsed()

    # Precompute all triples in lex order
    n = 34
    triples = list(combinations(range(n), 3))
    m = len(triples)  # C(34,3) = 5984

    # Map triple -> index
    triple_to_idx = {}
    for i, (a, b, c) in enumerate(triples):
        triple_to_idx[(a, b, c)] = i

    # For each vertex, list of triple indices containing it
    vertex_triples = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(triples):
        vertex_triples[a].append(i)
        vertex_triples[b].append(i)
        vertex_triples[c].append(i)

    # Precompute: for each 4-set, the 4 triples (to detect red K4^3)
    # For each 5-set, the 10 triples (to detect blue K5^3)
    # We'll track counts dynamically.

    # For red K4^3: for each 4-set S, count how many of its 4 triples are red.
    # For blue K5^3: for each 5-set S, count how many of its 10 triples are blue.

    # Precompute all 4-sets and their constituent triples
    four_sets = list(combinations(range(n), 4))
    four_set_triples = []
    for s in four_sets:
        t = [triple_to_idx[tup] for tup in combinations(s, 3)]
        four_set_triples.append(t)

    # Precompute all 5-sets and their constituent triples
    five_sets = list(combinations(range(n), 5))
    five_set_triples = []
    for s in five_sets:
        t = [triple_to_idx[tup] for tup in combinations(s, 3)]
        five_set_triples.append(t)

    # Initial coloring: random with slight bias
    # Use a structured initial: color based on modular arithmetic to start near-valid
    colors = np.zeros(m, dtype=np.int8)
    for i, (a, b, c) in enumerate(triples):
        # Use a combination of modular patterns for a good starting point
        s = a + b + c
        colors[i] = 0 if (s % 3 == 0 or s % 7 == 0) else 1

    # Count red triples in each 4-set and blue triples in each 5-set
    # red_count[j] = number of red triples in 4-set j
    # blue_count[j] = number of blue triples in 5-set j
    red_count = np.zeros(len(four_sets), dtype=np.int32)
    blue_count = np.zeros(len(five_sets), dtype=np.int32)

    for j, t in enumerate(four_set_triples):
        for idx in t:
            if colors[idx] == 0:
                red_count[j] += 1

    for j, t in enumerate(five_set_triples):
        for idx in t:
            if colors[idx] == 1:
                blue_count[j] += 1

    # Violation: red_count[j] == 4 or blue_count[j] == 10
    def total_violations():
        return int(np.sum(red_count == 4)) + int(np.sum(blue_count == 10))

    current_violations = total_violations()
    best_colors = colors.copy()
    best_violations = current_violations

    # For fast updates: for each triple index, which 4-sets and 5-sets contain it
    triple_to_four_sets = [[] for _ in range(m)]
    for j, t in enumerate(four_set_triples):
        for idx in t:
            triple_to_four_sets[idx].append(j)

    triple_to_five_sets = [[] for _ in range(m)]
    for j, t in enumerate(five_set_triples):
        for idx in t:
            triple_to_five_sets[idx].append(j)

    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9995

    step = 0
    rng = random.Random(42)

    while elapsed() < deadline:
        # Check time
        if time_left() < 0.5:
            break

        # Pick a random triple to flip
        idx = rng.randint(0, m - 1)
        old_color = colors[idx]
        new_color = 1 - old_color

        # Compute delta in violations
        delta_v = 0

        # Update red counts for 4-sets containing this triple
        for j in triple_to_four_sets[idx]:
            if old_color == 0:
                # Was red, now blue: red_count decreases
                if red_count[j] == 4:
                    delta_v -= 1  # remove a violation
                red_count[j] -= 1
                if red_count[j] == 4:
                    delta_v += 1  # add a violation (can't happen since we decreased)
            else:
                # Was blue, now red: red_count increases
                if red_count[j] == 3:
                    delta_v += 1  # 3->4 creates violation
                red_count[j] += 1
                if red_count[j] == 4:
                    pass  # already counted above

        # Update blue counts for 5-sets containing this triple
        for j in triple_to_five_sets[idx]:
            if old_color == 1:
                # Was blue, now red: blue_count decreases
                if blue_count[j] == 10:
                    delta_v -= 1
                blue_count[j] -= 1
            else:
                # Was red, now blue: blue_count increases
                if blue_count[j] == 9:
                    delta_v += 1  # 9->10 creates violation
                blue_count[j] += 1

        new_violations = current_violations + delta_v

        # Accept if better or with probability exp(-delta/T)
        if new_violations < current_violations or (delta_v > 0 and rng.random() < np.exp(-delta_v / T)):
            colors[idx] = new_color
            current_violations = new_violations

            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                if best_violations == 0:
                    break
        else:
            # Revert
            for j in triple_to_four_sets[idx]:
                if old_color == 0:
                    red_count[j] += 1
                else:
                    red_count[j] -= 1
            for j in triple_to_five_sets[idx]:
                if old_color == 1:
                    blue_count[j] += 1
                else:
                    blue_count[j] -= 1

        step += 1
        T *= cooling
        if T < T_min:
            T = T_min

        # Periodic restart if stuck
        if step % 5000 == 0 and current_violations > 0:
            # Small perturbation: flip a few random triples
            for _ in range(5):
                idx2 = rng.randint(0, m - 1)
                old_c = colors[idx2]
                new_c = 1 - old_c
                colors[idx2] = new_c
                for j in triple_to_four_sets[idx2]:
                    if old_c == 0:
                        red_count[j] -= 1
                    else:
                        red_count[j] += 1
                for j in triple_to_five_sets[idx2]:
                    if old_c == 1:
                        blue_count[j] -= 1
                    else:
                        blue_count[j] += 1
            current_violations = total_violations()
            T = max(T, 0.1)

    # Use best found
    if best_violations == 0:
        final_colors = best_colors
    else:
        # Try to find any valid coloring; if none, return best effort
        # But we need to return a valid one or the score is 0
        # Let's do a final greedy repair attempt
        final_colors = best_colors.copy()
        # Recompute violations for final_colors
        rc = np.zeros(len(four_sets), dtype=np.int32)
        bc = np.zeros(len(five_sets), dtype=np.int32)
        for j, t in enumerate(four_set_triples):
            for idx in t:
                if final_colors[idx] == 0:
                    rc[j] += 1
        for j, t in enumerate(five_set_triples):
            for idx in t:
                if final_colors[idx] == 1:
                    bc[j] += 1

        # Greedy repair: fix violations one by one
        for _ in range(100):
            if time_left() < 1.0:
                break
            # Find a violating 4-set
            fixed = False
            viol_4 = np.where(rc == 4)[0]
            viol_5 = np.where(bc == 10)[0]
            if len(viol_4) > 0:
                j = viol_4[0]
                t = four_set_triples[j]
                # Flip a red triple in this set to blue
                for idx in t:
                    if final_colors[idx] == 0:
                        final_colors[idx] = 1
                        # Update counts
                        for j2 in triple_to_four_sets[idx]:
                            rc[j2] -= 1
                        for j2 in triple_to_five_sets[idx]:
                            bc[j2] += 1
                        fixed = True
                        break
            elif len(viol_5) > 0:
                j = viol_5[0]
                t = five_set_triples[j]
                for idx in t:
                    if final_colors[idx] == 1:
                        final_colors[idx] = 0
                        for j2 in triple_to_four_sets[idx]:
                            rc[j2] += 1
                        for j2 in triple_to_five_sets[idx]:
                            bc[j2] -= 1
                        fixed = True
                        break
            if not fixed:
                break

    # Convert to list of ints
    color_list = [int(c) for c in final_colors]

    # Verify validity
    def is_valid(colors_list, n):
        ct = {}
        for i, (a, b, c) in enumerate(triples):
            ct[(a, b, c)] = colors_list[i]
        # Check red K4^3
        for s in four_sets:
            count = 0
            for tup in combinations(s, 3):
                if ct[tup] == 0:
                    count += 1
            if count == 4:
                return False
        # Check blue K5^3
        for s in five_sets:
            count = 0
            for tup in combinations(s, 3):
                if ct[tup] == 1:
                    count += 1
            if count == 10:
                return False
        return True

    if not is_valid(color_list, n):
        # Try smaller n if n=34 fails
        for n_try in range(33, 5, -1):
            if time_left() < 2.0:
                break
            triples_t = list(combinations(range(n_try), 3))
            m_t = len(triples_t)
            ct_t = {}
            for i, (a, b, c) in enumerate(triples_t):
                # Re-derive from final_colors for smaller n (use first m_t entries? No, need consistent)
                # Just use modular coloring as fallback
                ct_t[(a, b, c)] = 0 if (a + b + c) % 3 == 0 else 1
            # Check
            valid = True
            for s in combinations(range(n_try), 4):
                count = sum(1 for tup in combinations(s, 3) if ct_t[tup] == 0)
                if count == 4:
                    valid = False
                    break
            if valid:
                for s in combinations(range(n_try), 5):
                    count = sum(1 for tup in combinations(s, 3) if ct_t[tup] == 1)
                    if count == 10:
                        valid = False
                        break
            if valid:
                return {'n': n_try, 'colors': [ct_t[tup] for tup in triples_t]}

    return {'n': n, 'colors': color_list}
# EVOLVE-BLOCK-END
