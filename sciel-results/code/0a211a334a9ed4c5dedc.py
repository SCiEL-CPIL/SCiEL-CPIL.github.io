"""Multi-Island Differential Evolution for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-island Differential Evolution with error-repair for multimodal BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(12345)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            try:
                f = func(x)
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Multi-Island DE Configuration ---
        n_islands = 4
        pop_per_island = 10
        total_pop = n_islands * pop_per_island  # 40 individuals
        F_base = 0.8  # Mutation scale factor
        CR = 0.9      # Crossover rate
        stagnation_limit = 15  # Generations without island improvement before restart
        max_generations = 80

        # Initialize islands with diverse starting points
        islands = []
        for i in range(n_islands):
            # Each island starts from a different region
            center = rng.uniform(-4.0, 4.0, size=dim)
            members = []
            for j in range(pop_per_island):
                if j == 0:
                    x = center.copy()
                else:
                    x = center + rng.normal(0, 1.5, size=dim)
                    x = np.clip(x, -5.0, 5.0)
                f = do_call(x)
                if f is None:
                    return best_x, best_f
                members.append({'x': x.copy(), 'f': f})
            # Sort by fitness
            members.sort(key=lambda m: m['f'])
            islands.append({
                'members': members,
                'best_f': members[0]['f'],
                'best_x': members[0]['x'].copy(),
                'stagnation': 0,
                'prev_best': members[0]['f']
            })

        # Track global best across islands
        global_best_f = min(isl['best_f'] for isl in islands)
        global_best_x = min(islands, key=lambda isl: isl['best_f'])['best_x'].copy()

        # Error repair counter for the open question
        repair_count = 0
        repair_improved_island = 0

        # --- Main DE Loop ---
        for gen in range(max_generations):
            if call_count >= budget or time.time() > deadline:
                break

            # Adaptive F: decrease slightly over time for convergence
            F = F_base * (0.7 ** (gen / 20.0)) + 0.3
            F = min(F, 1.0)

            for isl_idx in range(n_islands):
                if call_count >= budget or time.time() > deadline:
                    break

                isl = islands[isl_idx]
                members = isl['members']
                n = len(members)

                # Select parents from this island and potentially others
                # DE/rand/1 with occasional cross-island selection
                for i in range(n):
                    if call_count >= budget or time.time() > deadline:
                        break

                    # Choose three distinct parents (j, k != i)
                    candidates = list(range(n))
                    candidates.remove(i)
                    if len(candidates) < 2:
                        continue
                    j, k = rng.choice(candidates, size=2, replace=False)

                    # Also occasionally pull from another island for diversity
                    if rng.random() < 0.15 and n_islands > 1:
                        other_isl = islands[(isl_idx + 1 + rng.integers(n_islands - 1)) % n_islands]
                        if len(other_isl['members']) > 0:
                            k = -1  # Mark for cross-island
                            k_member = other_isl['members'][rng.integers(len(other_isl['members']))]
                    else:
                        k_member = None

                    x_i = members[i]['x']
                    x_j = members[j]['x']

                    if k == -1 and k_member is not None:
                        x_k = k_member['x']
                    else:
                        x_k = members[k]['x']

                    # Mutation: v = x_i + F * (x_j - x_k)
                    v = x_i + F * (x_j - x_k)

                    # Error repair: clip to bounds, handle non-finite
                    v = np.clip(v, -5.0, 5.0)
                    if not np.all(np.isfinite(v)):
                        # Repair: replace non-finite with best island member
                        v = isl['best_x'].copy() + rng.normal(0, 0.5, size=dim)
                        v = np.clip(v, -5.0, 5.0)
                        repair_count += 1

                    # Evaluate mutant
                    f_v = do_call(v)
                    if f_v is None:
                        break

                    # Track if repair led to island improvement
                    if repair_count > 0 and f_v < isl['best_f']:
                        repair_improved_island += 1

                    # Selection: replace if better
                    if f_v < members[i]['f']:
                        members[i]['x'] = v.copy()
                        members[i]['f'] = f_v

                # Re-sort island members
                members.sort(key=lambda m: m['f'])
                new_best_f = members[0]['f']
                new_best_x = members[0]['x'].copy()

                # Update island best
                if new_best_f < isl['best_f']:
                    isl['best_f'] = new_best_f
                    isl['best_x'] = new_best_x
                    isl['stagnation'] = 0
                else:
                    isl['stagnation'] += 1

                # Update global best
                if new_best_f < global_best_f:
                    global_best_f = new_best_f
                    global_best_x = new_best_x.copy()

                # Restart stagnant island
                if isl['stagnation'] >= stagnation_limit:
                    # Reset island: keep best, re-seed rest around global best + noise
                    best_member = members[0]
                    for j in range(1, n):
                        x_new = global_best_x + rng.normal(0, 2.0, size=dim)
                        x_new = np.clip(x_new, -5.0, 5.0)
                        f_new = do_call(x_new)
                        if f_new is None:
                            break
                        members[j] = {'x': x_new.copy(), 'f': f_new}
                    if call_count >= budget:
                        break
                    members.sort(key=lambda m: m['f'])
                    isl['members'] = members
                    isl['best_f'] = members[0]['f']
                    isl['best_x'] = members[0]['x'].copy()
                    isl['stagnation'] = 0

            # Periodic migration: swap worst member of each island with best of neighbor
            if gen % 5 == 4 and call_count < budget:
                for i in range(n_islands):
                    src = (i + 1) % n_islands
                    islands[i]['members'][-1] = islands[src]['members'][0].copy()
                    # Re-sort
                    islands[i]['members'].sort(key=lambda m: m['f'])
                    islands[i]['best_f'] = islands[i]['members'][0]['f']
                    islands[i]['best_x'] = islands[i]['members'][0]['x'].copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
