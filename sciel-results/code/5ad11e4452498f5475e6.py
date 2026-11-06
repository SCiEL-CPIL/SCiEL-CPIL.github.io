"""Edge-colourings of K16: minimize colours so every K4 sees >=5 colours."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx

    # Precompute all K4s and their edge index lists
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        e_list = [edge_idx[(a, b)] for a, b in combinations(k4, 2)]
        k4_edges.append(np.array(e_list, dtype=np.int32))
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)

    # For each edge, precompute which K4s contain it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in edge_to_k4s]

    def count_distinct(c):
        return len(np.unique(c))

    def check_violations(c):
        """Return number of K4s with < 5 distinct colors."""
        colors_in_k4 = c[k4_edges_arr]  # (1820, 6)
        sorted_c = np.sort(colors_in_k4, axis=1)
        distinct = 1 + np.sum(sorted_c[:, 1:] != sorted_c[:, :-1], axis=1)
        return np.sum(distinct < 5)

    def is_valid(c):
        return check_violations(c) == 0

    # Initial construction: try several modular schemes, pick the one with fewest colors
    best_init = None
    best_init_count = 100
    
    for mod in [13, 14, 15, 16, 17]:
        c = np.array([(i + j) % mod for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if is_valid(c):
            cnt = count_distinct(c)
            if best_init is None or cnt < best_init_count:
                best_init = c.copy()
                best_init_count = cnt
            if cnt <= 13:
                break
    
    if best_init is None:
        # Fallback: random valid coloring with more colors
        best_init = np.array([(i * 7 + j * 3) % 19 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if not is_valid(best_init):
            best_init = np.array([(i + j) % 20 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        best_init_count = count_distinct(best_init)

    best_colors = best_init.copy()
    best_count = best_init_count

    # Aggressive color merging: try to merge pairs of colors
    def try_merge(c, color_a, color_b):
        """Try to merge color_b into color_a."""
        if color_a == color_b:
            return None
        new_c = c.copy()
        new_c[new_c == color_b] = color_a
        if is_valid(new_c):
            return new_c
        return None

    # Iterative merging phase
    for iteration in range(1000):
        if time.time() > deadline - 40:
            break

        current_colors = np.unique(best_colors)
        if len(current_colors) <= 13:
            break

        # Count frequency
        color_freq = np.bincount(best_colors, minlength=120)
        used_colors = np.where(color_freq > 0)[0]

        # Sort by frequency (rarest first)
        sorted_colors = used_colors[np.argsort(color_freq[used_colors])]

        merged = False
        for target in sorted_colors:
            if time.time() > deadline - 40:
                break
            if color_freq[target] == 0:
                continue
            for donor in sorted_colors:
                if donor == target:
                    continue
                if color_freq[donor] == 0:
                    continue
                new_c = try_merge(best_colors, donor, target)
                if new_c is not None:
                    best_colors = new_c
                    best_count = count_distinct(best_colors)
                    merged = True
                    break
            if merged:
                break

        if not merged:
            # Try random edge reassignment to enable future merges
            existing = np.unique(best_colors)
            if len(existing) > 0:
                num_reassign = random.randint(1, 8)
                new_c = best_colors.copy()
                for _ in range(num_reassign):
                    edge_pos = random.randint(0, num_edges - 1)
                    new_c[edge_pos] = random.choice(existing)
                if is_valid(new_c):
                    best_colors = new_c
                    best_count = count_distinct(best_colors)

    # Improved simulated annealing with better acceptance criteria
    def annealing(c, start_temp, end_temp, steps):
        current = c.copy()
        current_count = count_distinct(current)
        current_violations = check_violations(current)
        best = current.copy()
        best_count = current_count
        best_violations = current_violations

        for step in range(steps):
            if time.time() > deadline - 10:
                break
            temp = start_temp * (end_temp / start_temp) ** (step / steps)

            # Pick a random edge and change its color
            edge_pos = random.randint(0, num_edges - 1)
            old_color = current[edge_pos]
            
            # Pick a random new color from existing colors
            existing = np.unique(current)
            if len(existing) > 0:
                new_color = random.choice(existing)
            else:
                new_color = 0

            current[edge_pos] = new_color
            new_count = count_distinct(current)
            new_violations = check_violations(current)

            # Energy: prefer fewer colors, penalize violations heavily
            energy_new = new_count * 10 + new_violations * 1000
            energy_old = current_count * 10 + current_violations * 1000

            # Acceptance: always accept if better, accept worse with probability exp(-delta/T)
            delta = energy_new - energy_old
            if delta < 0 or (delta == 0 and new_violations <= current_violations) or \
               (new_violations == 0 and random.random() < np.exp(-delta / max(temp, 1e-10))):
                current_count = new_count
                current_violations = new_violations
                if new_violations == 0 and new_count < best_count:
                    best = current.copy()
                    best_count = new_count
                    best_violations = new_violations
            else:
                # Reject
                current[edge_pos] = old_color

        return best, best_count

    # Run multiple annealing restarts if we have time
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        if best_count <= 13:
            break
        
        # Create a perturbed starting point
        if restart == 0:
            start_c = best_colors.copy()
        else:
            start_c = best_colors.copy()
            # Perturb some edges
            num_perturb = random.randint(5, 15)
            existing = np.unique(start_c)
            for _ in range(num_perturb):
                edge_pos = random.randint(0, num_edges - 1)
                start_c[edge_pos] = random.choice(existing)
            if not is_valid(start_c):
                # Revert to valid state
                start_c = best_colors.copy()
        
        # Run annealing
        new_best, new_count = annealing(start_c, 3.0, 0.01, 30000)
        
        if new_count < best_count:
            best_colors = new_best
            best_count = new_count
    
    # Final merge pass
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        current_colors = np.unique(best_colors)
        if len(current_colors) <= 13:
            break
        color_freq = np.bincount(best_colors, minlength=120)
        used_colors = np.where(color_freq > 0)[0]
        sorted_colors = used_colors[np.argsort(color_freq[used_colors])]
        merged = False
        for target in sorted_colors:
            if time.time() > deadline - 5:
                break
            if color_freq[target] == 0:
                continue
            for donor in sorted_colors:
                if donor == target or color_freq[donor] == 0:
                    continue
                new_c = try_merge(best_colors, donor, target)
                if new_c is not None:
                    best_colors = new_c
                    best_count = count_distinct(best_colors)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    # Final validity check
    if not is_valid(best_colors):
        # Fallback
        best_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    return best_colors.tolist()
