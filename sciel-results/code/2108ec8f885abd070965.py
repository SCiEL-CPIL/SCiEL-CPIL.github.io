import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def solve_n(n):
        """
        Attempts to find a valid coloring for n vertices.
        Returns list of colors (0 for red, 1 for blue) for all triples in lexicographic order.
        Returns None if not found within time limit.
        """
        if n < 1:
            return None

        # Generate all triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))

        # Precompute incidence lists
        # For each triple, which K4s and K5s does it belong to?
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        # For each K4, list of triple indices
        k4_triples = []
        for k4 in k4s:
            t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k4s[idx].append(len(k4_triples) - 1)

        # For each K5, list of triple indices
        k5_triples = []
        for k5 in k5s:
            t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5_triples.append(t_indices)
            for idx in t_indices:
                triple_to_k5s[idx].append(len(k5_triples) - 1)

        num_k4s = len(k4s)
        num_k5s = len(k5s)

        # Local Search with Restarts
        max_restarts = 30
        restart_time_limit = 4.0  # seconds per restart

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(num_triples)]

            # Initialize red counts
            rc_k4 = [0] * num_k4s
            rc_k5 = [0] * num_k5s

            for i, c in enumerate(colors):
                if c == 0:  # Red
                    for k4_idx in triple_to_k4s[i]:
                        rc_k4[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        rc_k5[k5_idx] += 1

            # Count initial violations
            violations = 0
            for r in rc_k4:
                if r == 4:
                    violations += 1
            for r in rc_k5:
                if r == 0:
                    violations += 1

            if violations == 0:
                return colors

            # Simulated Annealing parameters
            T = 1.0
            cooling = 0.999
            min_T = 0.01

            current_time = time.time()
            steps = 0
            max_steps = 50000

            while steps < max_steps and time.time() < current_time + restart_time_limit:
                if time.time() > deadline - 5:
                    break

                # Pick random triple
                t = random.randint(0, num_triples - 1)
                old_color = colors[t]
                new_color = 1 - old_color

                delta = 0

                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    if old_color == 0:  # Red -> Blue
                        # rc_k4 decreases by 1
                        if rc_k4[k4_idx] == 4:
                            delta -= 1  # Violation removed
                    else:  # Blue -> Red
                        # rc_k4 increases by 1
                        if rc_k4[k4_idx] == 3:
                            delta += 1  # Violation added

                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    if old_color == 0:  # Red -> Blue
                        # rc_k5 decreases by 1
                        if rc_k5[k5_idx] == 0:
                            delta -= 1  # Violation removed (was all blue, now not)
                    else:  # Blue -> Red
                        # rc_k5 increases by 1
                        if rc_k5[k5_idx] == 0:
                            delta -= 1  # Violation removed (was all blue, now not)

                # Accept move
                if delta < 0 or (random.random() < pow(2.718, -delta / max(T, 0.001))):
                    # Apply move
                    colors[t] = new_color

                    # Update counts
                    for k4_idx in triple_to_k4s[t]:
                        if old_color == 0:
                            rc_k4[k4_idx] -= 1
                        else:
                            rc_k4[k4_idx] += 1

                    for k5_idx in triple_to_k5s[t]:
                        if old_color == 0:
                            rc_k5[k5_idx] -= 1
                        else:
                            rc_k5[k5_idx] += 1

                    violations += delta

                    if violations == 0:
                        return colors

                    T *= cooling
                    if T < min_T:
                        T = min_T
                else:
                    T *= cooling
                    if T < min_T:
                        T = min_T
                steps += 1

        return None

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 15:
            break
        colors = solve_n(n)
        if colors is not None:
            return {'n': n, 'colors': colors}

    # Fallback: return seed result
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
