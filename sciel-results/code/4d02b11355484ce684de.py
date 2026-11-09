import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            if len(x) != dim:
                return None
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim, base_start=2):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        # Phase 1: Halton space-filling for systematic early exploration
        n_halton = 60
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts = halton_pts * 10.0 - 5.0  # Map to [-5, 5]

        # Add center point
        center = np.zeros(dim)
        all_points = np.vstack([center, halton_pts])

        for i in range(len(all_points)):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            fx = eval_func(all_points[i])
            if fx is None:
                break

        # Phase 2: Multi-fidelity (1+1)-ES with dynamic budget allocation
        # 5 trajectories with different step sizes
        base_sigmas = [0.1, 0.5, 1.0, 2.0, 4.0]
        n_trajectories = len(base_sigmas)
        remaining = budget - calls_made

        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Initialize trajectories from the best point found so far
        trajectories = []
        for i in range(n_trajectories):
            x = best_x.copy()
            sigma = base_sigmas[i]
            fx = best_f
            recent_improvements = []
            trajectories.append({
                'x': x,
                'sigma': sigma,
                'fx': fx,
                'recent_improvements': recent_improvements,
                'calls': 0
            })

        # Dynamic budget allocation: allocate to trajectories with best recent improvement
        while calls_made < budget and time.time() < deadline:
            # Calculate allocation weights based on recent improvements
            weights = np.ones(n_trajectories)
            for i, traj in enumerate(trajectories):
                if len(traj['recent_improvements']) >= 5:
                    # Reward trajectories that have been improving recently
                    recent = traj['recent_improvements'][-5:]
                    if any(imp > 1e-8 for imp in recent):
                        weights[i] = 2.0
                    elif any(imp > 1e-6 for imp in recent):
                        weights[i] = 1.5

            # Normalize weights
            total_weight = np.sum(weights)
            if total_weight == 0:
                weights = np.ones(n_trajectories)
                total_weight = n_trajectories

            # Allocate next batch of calls (batch size 5)
            batch_size = min(5, budget - calls_made)
            if batch_size <= 0:
                break

            # Distribute calls proportionally
            raw_allocation = (weights / total_weight) * batch_size
            allocation = np.floor(raw_allocation).astype(int)
            # Distribute remaining calls to trajectories with largest fractional parts
            remaining_calls = batch_size - np.sum(allocation)
            if remaining_calls > 0:
                fracs = raw_allocation - allocation
                for _ in range(remaining_calls):
                    idx = np.argmax(fracs)
                    allocation[idx] += 1
                    fracs[idx] = -1  # Mark as assigned

            # Execute allocated calls
            for i in range(n_trajectories):
                for _ in range(allocation[i]):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    traj = trajectories[i]
                    rng = np.random.default_rng(int(time.time() * 1000) % 1000000)
                    y = rng.standard_normal(dim)
                    x_trial = traj['x'] + traj['sigma'] * y
                    x_trial = np.clip(x_trial, -5.0, 5.0)
                    f_trial = eval_func(x_trial)

                    if f_trial is None:
                        break

                    improvement = traj['fx'] - f_trial
                    traj['recent_improvements'].append(improvement)
                    if len(traj['recent_improvements']) > 20:
                        traj['recent_improvements'].pop(0)

                    if f_trial < traj['fx']:
                        traj['x'] = x_trial
                        traj['fx'] = f_trial
                        traj['sigma'] *= 1.05
                    else:
                        traj['sigma'] *= 0.95

                    # Adapt sigma based on recent success rate
                    if len(traj['recent_improvements']) >= 10:
                        recent = traj['recent_improvements'][-10:]
                        success_rate = np.sum(np.array(recent) > 1e-8) / len(recent)
                        target_success = 1.0 / (dim + 1)
                        if success_rate > target_success * 1.5:
                            traj['sigma'] *= 1.1
                        elif success_rate < target_success * 0.5:
                            traj['sigma'] *= 0.9

                    traj['sigma'] = np.clip(traj['sigma'], 1e-6, 5.0)
                    traj['calls'] += 1

        # Phase 3: Final Nelder-Mead polish on the best point
        if calls_made < budget - 10 and best_x is not None and time.time() < deadline:
            x0 = best_x.copy()
            n = dim
            rng = np.random.default_rng(42)

            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.1 * max(0.1, abs(x0[i]))

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                fx = eval_func(simplex[i])
                if fx is None:
                    break
                fvals[i] = fx

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_nm_iter = min(20, (budget - calls_made) // (n + 1))
            for iteration in range(max_nm_iter):
                if time.time() > deadline or calls_made >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                if calls_made >= budget or time.time() > deadline:
                    break
                f_reflect = eval_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    if calls_made >= budget or time.time() > deadline:
                        break
                    f_expand = eval_func(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < fvals[-1]:
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    if calls_made >= budget or time.time() > deadline:
                        break
                    f_contract = eval_func(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= budget or time.time() > deadline:
                                break
                            fx = eval_func(simplex[i])
                            if fx is None:
                                break
                            fvals[i] = fx
                        break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

        return best_x, best_f
