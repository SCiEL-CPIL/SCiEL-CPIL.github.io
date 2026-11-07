"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex optimizer with adaptive restarts.

    Strategy: Use a population of Nelder-Mead simplex agents. NM is very
    efficient in low dimensions (5D) and handles rotated landscapes well.
    We run multiple independent NM instances in parallel (interleaved),
    each with its own initial simplex. When an agent stagnates, it restarts
    near the global best or at a fresh random location.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        # ---- Nelder-Mead simplex implementation --------------------------
        # Each agent has a simplex of dim+1 points.
        # We interleave evaluations across agents.

        def nm_init(center, step=1.0):
            """Create initial simplex around center."""
            simplex = []
            # First vertex is the center
            simplex.append(center.copy())
            # Other vertices are center + step * basis vectors
            for i in range(dim):
                v = center.copy()
                v[i] += step
                simplex.append(v)
            return simplex

        def nm_evaluate(simplex):
            """Evaluate all points in simplex, return list of (x, f)."""
            return [(x, func(x)) for x in simplex]

        def nm_step(simplex, values, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """
            Perform one Nelder-Mead step. Returns new simplex and which index
            to re-evaluate (the one that was replaced), or None if no change.
            Actually, we'll just return the new simplex and let the caller
            evaluate.
            """
            n = len(simplex)
            # Sort by function value
            order = np.argsort(values)
            sorted_simplex = [simplex[i] for i in order]
            sorted_values = [values[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(sorted_simplex[:-1], axis=0)

            # Reflection
            x_worst = sorted_simplex[-1]
            x_reflect = centroid + alpha * (centroid - x_worst)

            return x_reflect, centroid, sorted_simplex, sorted_values

        # ---- Initialize multiple NM agents --------------------------------
        n_agents = 8
        agents = []

        # First, get some initial evaluations to set up agents
        # Start with diverse random points
        initial_points = []
        for _ in range(n_agents):
            p = rng.uniform(-5.0, 5.0, dim)
            initial_points.append(p)

        # Evaluate initial points
        for p in initial_points:
            f = func(p)
            record(p, f)
            if n_calls >= budget or time.time() > deadline:
                break

        if n_calls >= budget or time.time() > deadline:
            return x_best, f_best

        # Create NM agents, each starting from one of the initial points
        for i in range(n_agents):
            center = initial_points[i]
            simplex = nm_init(center, step=1.0)
            # We already have the value for the center (first point of simplex)
            # Need to evaluate the other dim points
            values = [None] * (dim + 1)
            values[0] = func(simplex[0]) if n_calls < budget else np.inf
            record(simplex[0], values[0]) if values[0] != np.inf else None

            # Evaluate remaining simplex points
            for j in range(1, dim + 1):
                if n_calls >= budget or time.time() > deadline:
                    values[j] = np.inf
                    break
                values[j] = func(simplex[j])
                record(simplex[j], values[j])

            agents.append({
                'simplex': simplex,
                'values': values,
                'stall': 0,
                'best_local': min(v for v in values if v != np.inf) if any(v != np.inf for v in values) else np.inf
            })

        # ---- Main loop: interleave NM steps across agents ----------------
        cursor = 0
        while n_calls < budget and time.time() < deadline:
            a = agents[cursor % n_agents]
            cursor += 1

            simplex = a['simplex']
            values = a['values']

            # Check if simplex is still valid (all values are finite)
            if any(v is None or v == np.inf for v in values):
                # Re-initialize this agent near global best
                center = x_best + rng.standard_normal(dim) * 0.5
                center = np.clip(center, -5.0, 5.0)
                simplex = nm_init(center, step=0.5)
                values = [None] * (dim + 1)
                for j in range(dim + 1):
                    if n_calls >= budget or time.time() > deadline:
                        values[j] = np.inf
                        break
                    values[j] = func(simplex[j])
                    record(simplex[j], values[j])
                a['simplex'] = simplex
                a['values'] = values
                a['stall'] = 0
                continue

            # Perform NM step
            x_reflect, centroid, sorted_simplex, sorted_values = nm_step(simplex, values)

            # Evaluate reflection point
            if n_calls >= budget or time.time() > deadline:
                break

            f_reflect = func(x_reflect)
            record(x_reflect, f_reflect)

            f_worst = sorted_values[-1]
            f_best_local = sorted_values[0]

            improved = False

            if f_reflect < f_worst:
                # Accept reflection
                new_simplex = sorted_simplex[:-1] + [x_reflect]
                new_values = sorted_values[:-1] + [f_reflect]

                # Check for expansion
                if f_reflect < f_best_local:
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    if n_calls >= budget or time.time() > deadline:
                        # Can't expand, just accept reflection
                        a['simplex'] = new_simplex
                        a['values'] = new_values
                        a['stall'] = 0
                        a['best_local'] = min(a['best_local'], f_reflect)
                        continue

                    f_expand = func(x_expand)
                    record(x_expand, f_expand)

                    if f_expand < f_reflect:
                        new_simplex = sorted_simplex[:-1] + [x_expand]
                        new_values = sorted_values[:-1] + [f_expand]
                        improved = True
                    else:
                        improved = True
                else:
                    improved = True

                a['simplex'] = new_simplex
                a['values'] = new_values
                a['stall'] = 0
                a['best_local'] = min(a['best_local'], min(new_values))

            else:
                # Contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = centroid + 0.5 * (x_reflect - centroid)
                else:
                    # Inside contraction
                    x_contract = centroid - 0.5 * (x_worst - centroid)

                if n_calls >= budget or time.time() > deadline:
                    break

                f_contract = func(x_contract)
                record(x_contract, f_contract)

                if f_contract < f_worst:
                    new_simplex = sorted_simplex[:-1] + [x_contract]
                    new_values = sorted_values[:-1] + [f_contract]
                    a['simplex'] = new_simplex
                    a['values'] = new_values
                    a['stall'] = 0
                    a['best_local'] = min(a['best_local'], f_contract)
                else:
                    # Shrink: move all points toward best
                    x_best_simplex = sorted_simplex[0]
                    new_simplex = [x_best_simplex + 0.5 * (x - x_best_simplex) for x in simplex]
                    # Re-evaluate all points
                    new_values = []
                    for x in new_simplex:
                        if n_calls >= budget or time.time() > deadline:
                            new_values.append(np.inf)
                            break
                        f = func(x)
                        record(x, f)
                        new_values.append(f)

                    if all(v != np.inf for v in new_values):
                        a['simplex'] = new_simplex
                        a['values'] = new_values
                        a['stall'] += 1
                    else:
                        # Reset agent
                        center = x_best + rng.standard_normal(dim) * 0.3
                        center = np.clip(center, -5.0, 5.0)
                        simplex = nm_init(center, step=0.3)
                        values = [None] * (dim + 1)
                        for j in range(dim + 1):
                            if n_calls >= budget or time.time() > deadline:
                                values[j] = np.inf
                                break
                            values[j] = func(simplex[j])
                            record(simplex[j], values[j])
                        a['simplex'] = simplex
                        a['values'] = values
                        a['stall'] = 0
                        continue

            # Restart if stalled too much
            if a['stall'] >= 30:
                if rng.random() < 0.5:
                    # Restart near global best
                    center = x_best + rng.standard_normal(dim) * 0.5
                else:
                    # Fresh random restart
                    center = rng.uniform(-5.0, 5.0, dim)

                center = np.clip(center, -5.0, 5.0)
                simplex = nm_init(center, step=1.0)
                values = [None] * (dim + 1)
                for j in range(dim + 1):
                    if n_calls >= budget or time.time() > deadline:
                        values[j] = np.inf
                        break
                    values[j] = func(simplex[j])
                    record(simplex[j], values[j])

                a['simplex'] = simplex
                a['values'] = values
                a['stall'] = 0
                a['best_local'] = min(v for v in values if v != np.inf) if any(v != np.inf for v in values) else np.inf

        return x_best, f_best
# EVOLVE-BLOCK-END
