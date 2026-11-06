import time
import random
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]

    for n in range(34, 9, -1):
        if time.time() > deadline - 25.0:
            break

        try:
            edges = list(itertools.combinations(range(n), 3))
            num_edges = len(edges)
            edge_index = {e: i for i, e in enumerate(edges)}

            sets4 = list(itertools.combinations(range(n), 4))
            sets5 = list(itertools.combinations(range(n), 5))
            num_sets4 = len(sets4)
            num_sets5 = len(sets5)

            # Precompute: for each 4-set, which 3 edges are red-critical
            # For each 5-set, which 10 edges are blue-critical
            # edge -> list of 4-set indices
            edge_to_4sets = [[] for _ in range(num_edges)]
            for s4_idx, s4 in enumerate(sets4):
                for i in range(4):
                    triple = s4[:i] + s4[i+1:]
                    e_idx = edge_index[triple]
                    edge_to_4sets[e_idx].append(s4_idx)

            # edge -> list of 5-set indices
            edge_to_5sets = [[] for _ in range(num_edges)]
            for s5_idx, s5 in enumerate(sets5):
                for triple in itertools.combinations(s5, 3):
                    e_idx = edge_index[triple]
                    edge_to_5sets[e_idx].append(s5_idx)

        except Exception:
            continue

        found = False
        max_restarts = 12

        for restart in range(max_restarts):
            if time.time() > deadline - 12.0:
                break

            # Structured initialization: use algebraic seed to reduce initial violations
            # Use a linear function over GF(2) for initial coloring
            colors = np.zeros(num_edges, dtype=np.int32)
            for i, (a, b, c) in enumerate(edges):
                # Simple algebraic rule: color based on parity of vertex indices
                colors[i] = (a * 7 + b * 11 + c * 13) % 2

            # Compute initial violation counts
            red_count_4 = np.zeros(num_sets4, dtype=np.int32)
            blue_count_5 = np.zeros(num_sets5, dtype=np.int32)

            for e_idx in range(num_edges):
                c = colors[e_idx]
                if c == 0:
                    for s4_idx in edge_to_4sets[e_idx]:
                        red_count_4[s4_idx] += 1
                else:
                    for s5_idx in edge_to_5sets[e_idx]:
                        blue_count_5[s5_idx] += 1

            viol4_indices = np.where(red_count_4 == 4)[0]
            viol5_indices = np.where(blue_count_5 == 10)[0]
            total_viol = len(viol4_indices) + len(viol5_indices)

            if total_viol == 0:
                found = True
                break

            # Constraint-directed local search with forward checking
            # Phase 1: Greedy violation resolution
            # Phase 2: Simulated annealing refinement
            max_steps = 50000
            step = 0
            T = 0.5
            T_min = 0.005
            cooling = 0.9995

            while step < max_steps and total_viol > 0:
                if time.time() > deadline - 5.0:
                    break

                step += 1

                if T < T_min:
                    T = T_min * 2.0  # Reheat

                # Forward checking: find edges that are part of violating structures
                # Priority: edges in red K4 violations, then blue K5 violations
                candidate_edges = set()

                # Sample from violating 4-sets (if too many, sample)
                if len(viol4_indices) > 0:
                    sample_size = min(len(viol4_indices), 50)
                    sampled_viol4 = random.sample(list(viol4_indices), sample_size)
                    for s4_idx in sampled_viol4:
                        s4 = sets4[s4_idx]
                        # All 4 edges in this 4-set are candidates
                        for i in range(4):
                            triple = s4[:i] + s4[i+1:]
                            e_idx = edge_index[triple]
                            candidate_edges.add(e_idx)

                if len(viol5_indices) > 0:
                    sample_size = min(len(viol5_indices), 30)
                    sampled_viol5 = random.sample(list(viol5_indices), sample_size)
                    for s5_idx in sampled_viol5:
                        s5 = sets5[s5_idx]
                        # Sample a few edges from this 5-set
                        all_triples_5 = list(itertools.combinations(s5, 3))
                        sample_edges = random.sample(all_triples_5, min(5, len(all_triples_5)))
                        for triple in sample_edges:
                            e_idx = edge_index[triple]
                            candidate_edges.add(e_idx)

                if not candidate_edges:
                    # Fallback: random edge
                    e_idx = random.randint(0, num_edges - 1)
                else:
                    e_idx = random.choice(list(candidate_edges))

                old_color = colors[e_idx]
                new_color = 1 - old_color

                # Compute delta in violations
                delta_viol = 0.0

                if old_color == 0:  # Red -> Blue
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c - 1
                        if old_c == 4:
                            delta_viol -= 1.0
                        if new_c == 4:
                            delta_viol += 1.0
                        red_count_4[s4_idx] = new_c
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c + 1
                        if old_c == 10:
                            delta_viol -= 1.0
                        if new_c == 10:
                            delta_viol += 1.0
                        blue_count_5[s5_idx] = new_c
                else:  # Blue -> Red
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c - 1
                        if old_c == 10:
                            delta_viol -= 1.0
                        if new_c == 10:
                            delta_viol += 1.0
                        blue_count_5[s5_idx] = new_c
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c + 1
                        if old_c == 4:
                            delta_viol -= 1.0
                        if new_c == 4:
                            delta_viol += 1.0
                        red_count_4[s4_idx] = new_c

                total_viol += delta_viol

                # Accept with simulated annealing probability
                if delta_viol < 0 or (T > 0 and random.random() < np.exp(-delta_viol / T)):
                    colors[e_idx] = new_color
                    if total_viol == 0:
                        found = True
                        break
                else:
                    # Rollback
                    if old_color == 0:
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] += 1
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] -= 1
                    else:
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] += 1
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] -= 1
                    total_viol -= delta_viol

                T *= cooling

                # Update violation index sets periodically
                if step % 500 == 0 and total_viol > 0:
                    viol4_indices = np.where(red_count_4 == 4)[0]
                    viol5_indices = np.where(blue_count_5 == 10)[0]

            if found:
                best_n = n
                best_colors = colors.tolist()
                break

        if found:
            break

    return {'n': best_n, 'colors': best_colors}
