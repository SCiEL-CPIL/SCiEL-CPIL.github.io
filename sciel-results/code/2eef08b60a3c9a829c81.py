import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Fallback valid small coloring
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    def search_for_n(n, time_limit, seed=0):
        """
        Local search for a valid coloring of n vertices.
        Uses incremental violation counting and greedy + random restarts.
        """
        if time.time() > start_time + time_limit:
            return None

        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Build incidence structures
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))

        quad_triples = []
        for quad in quads:
            idxs = [tri_to_idx[tuple(quad[j] for j in range(4) if j != i)] for i in range(4)]
            quad_triples.append(idxs)

        pent_triples = []
        for pent in pents:
            idxs = [tri_to_idx[t] for t in combinations(pent, 3)]
            pent_triples.append(idxs)

        # For each triple, which quads/pents contain it
        triple_to_quads = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)

        triple_to_pents = [[] for _ in range(M)]
        for pidx, tidxs in enumerate(pent_triples):
            for t in tidxs:
                triple_to_pents[t].append(pidx)

        n_quads = len(quads)
        n_pents = len(pents)

        rng = np.random.RandomState(seed)

        # Try multiple restarts
        max_restarts = 50
        per_restart_time = time_limit / max_restarts

        for restart in range(max_restarts):
            if time.time() > start_time + time_limit:
                break

            # Initialize with random or heuristic coloring
            if restart % 3 == 0:
                # Heuristic: sum mod 3
                colors = np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples], dtype=np.int8)
            elif restart % 3 == 1:
                # Random
                colors = rng.randint(0, 2, size=M).astype(np.int8)
            else:
                # Heuristic: sum mod 2
                colors = np.array([(a + b + c) % 2 for a, b, c in triples], dtype=np.int8)

            # Compute initial violation counts
            quad_red_counts = np.zeros(n_quads, dtype=np.int32)
            pent_blue_counts = np.zeros(n_pents, dtype=np.int32)

            for i in range(M):
                if colors[i] == 0:  # Red
                    for qidx in triple_to_quads[i]:
                        quad_red_counts[qidx] += 1
                else:  # Blue
                    for pidx in triple_to_pents[i]:
                        pent_blue_counts[pidx] += 1

            violations = int(np.sum(quad_red_counts == 4)) + int(np.sum(pent_blue_counts == 10))

            if violations == 0:
                return colors.tolist()

            # Local search phase
            local_deadline = start_time + time_limit
            steps = 0
            max_steps = 500000
            stuck_count = 0
            best_violations = violations
            best_colors = colors.copy()

            while time.time() < local_deadline and steps < max_steps:
                steps += 1

                # Pick a triple to flip
                # Strategy: pick from violating cliques if any, else random
                if violations > 0:
                    # Find violating quads and pents
                    viol_quads = np.where(quad_red_counts == 4)[0]
                    viol_pents = np.where(pent_blue_counts == 10)[0]

                    candidates = set()
                    for q in viol_quads:
                        for t in quad_triples[q]:
                            candidates.add(t)
                    for p in viol_pents:
                        for t in pent_triples[p]:
                            candidates.add(t)

                    if candidates:
                        t = list(candidates)[rng.randint(0, len(candidates))]
                    else:
                        t = rng.randint(0, M)
                else:
                    t = rng.randint(0, M)

                old_color = colors[t]
                new_color = 1 - old_color

                # Compute delta violations
                delta = 0

                if old_color == 0:  # Red -> Blue
                    # Quads: red count decreases by 1 for each quad containing t
                    for qidx in triple_to_quads[t]:
                        if quad_red_counts[qidx] == 4:
                            delta -= 1  # Violation removed
                    # Pents: blue count increases by 1 for each pent containing t
                    for pidx in triple_to_pents[t]:
                        if pent_blue_counts[pidx] == 9:
                            delta += 1  # Violation added
                else:  # Blue -> Red
                    # Quads: red count increases by 1
                    for qidx in triple_to_quads[t]:
                        if quad_red_counts[qidx] == 3:
                            delta += 1  # Violation added
                    # Pents: blue count decreases by 1
                    for pidx in triple_to_pents[t]:
                        if pent_blue_counts[pidx] == 10:
                            delta -= 1  # Violation removed

                # Acceptance criterion: accept if delta < 0, or with probability if delta > 0
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = rng.rand() < 0.5
                else:
                    # Simulated annealing style: accept with probability exp(-delta / T)
                    # T decreases over time
                    T = max(0.1, 1.0 * (1.0 - steps / max_steps))
                    accept = rng.rand() < np.exp(-delta / T)

                if accept:
                    # Apply the flip
                    colors[t] = new_color

                    if old_color == 0:  # Was Red, now Blue
                        for qidx in triple_to_quads[t]:
                            quad_red_counts[qidx] -= 1
                        for pidx in triple_to_pents[t]:
                            pent_blue_counts[pidx] += 1
                    else:  # Was Blue, now Red
                        for qidx in triple_to_quads[t]:
                            quad_red_counts[qidx] += 1
                        for pidx in triple_to_pents[t]:
                            pent_blue_counts[pidx] -= 1

                    violations += delta

                    if violations == 0:
                        return colors.tolist()

                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                        stuck_count = 0
                    else:
                        stuck_count += 1

                    # If stuck for too long, perturb
                    if stuck_count > 10000:
                        # Random perturbation: flip 10 random triples
                        for _ in range(10):
                            pt = rng.randint(0, M)
                            old_pt = colors[pt]
                            new_pt = 1 - old_pt
                            colors[pt] = new_pt
                            if old_pt == 0:
                                for qidx in triple_to_quads[pt]:
                                    quad_red_counts[qidx] -= 1
                                for pidx in triple_to_pents[pt]:
                                    pent_blue_counts[pidx] += 1
                            else:
                                for qidx in triple_to_quads[pt]:
                                    quad_red_counts[qidx] += 1
                                for pidx in triple_to_pents[pt]:
                                    pent_blue_counts[pidx] -= 1
                        # Recompute violations
                        violations = int(np.sum(quad_red_counts == 4)) + int(np.sum(pent_blue_counts == 10))
                        stuck_count = 0

        return None

    # Try increasing n
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        per_n_budget = min(remaining - 2.0, 25.0)
        if per_n_budget <= 0:
            break

        found = None
        # Try a few seeds for each n
        for r in range(3):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_n_budget / 3.0, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 13 + 7)
            if res is not None:
                found = res
                best_n = n
                best_colors = found
                break  # Found for this n, move to next n

    if best_n < 10:
        # Fallback to a known valid small coloring
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    return {'n': best_n, 'colors': best_colors}
