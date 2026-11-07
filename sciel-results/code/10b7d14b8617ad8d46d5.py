"""Rewritten optimizer for BBOB-style f13 (Sharp ridge) using multi-start strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start optimizer: 50 random initializations, top-5 selected, parallel (1+1)-ES with CSA."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global multi-start exploration
        n_starts = 50
        n_top = 5

        # Evaluate n_starts random points
        points = []
        values = []
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)

        values = np.array(values)
        points = np.array(points)

        # Select top n_top points
        top_indices = np.argsort(values)[:n_top]
        top_points = points[top_indices]
        top_values = values[top_indices]

        calls_used = n_starts
        remaining_budget = budget - calls_used

        if remaining_budget <= 0:
            best_idx = np.argmin(top_values)
            return top_points[best_idx], top_values[best_idx]

        # Phase 2: Run (1+1)-ES with CSA from each of the top 5 starting points
        # We use a dynamic scheduling: always work on the currently best state.
        # If it stagnates, switch to the next best.
        
        # Initialize ES states
        es_states = []
        for i in range(n_top):
            es_states.append({
                'x': top_points[i].copy(),
                'fx': top_values[i],
                'sigma': 1.0,
                'h_s': np.zeros(dim), # History for CSA
                'stagnation_count': 0,
                'last_improvement_call': calls_used
            })

        # Precompute CSA parameters
        # c_s, d_s, theta, chi_n
        c_s = 1 + 2 * min(0.5, (1 - 1) / (dim + 2)) # mu_eff=1 for (1+1)
        # Standard (1+1) parameters:
        # c_s = 1 + 2 * min(0.5, 0) = 1 ? No, usually c_s is around 0.5-1.
        # Let's use standard CMA-ES (1+1) parameters:
        # c_s = 1 + 2 * min(0.5, (mu_eff - 1) / (n + 2)) with mu_eff = 1 -> c_s = 1.
        # d_s = n + 2*c_s + 2*c_s^2/n
        # theta = 2 * (c_s + 2) / (n + 4)
        # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(32n^2))
        
        c_s = 1.0
        d_s = dim + 2 * c_s + 2 * (c_s ** 2) / dim
        theta = 2 * (c_s + 2) / (dim + 4)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (32 * dim ** 2))

        best_overall_x = es_states[0]['x'].copy()
        best_overall_fx = es_states[0]['fx']
        for s in es_states[1:]:
            if s['fx'] < best_overall_fx:
                best_overall_fx = s['fx']
                best_overall_x = s['x'].copy()

        stagnation_threshold = 20 # Steps without improvement before switching

        while calls_used < budget:
            # Select state to update
            # Sort states by fx
            sorted_states = sorted(es_states, key=lambda s: s['fx'])
            
            # Find the best state that is not stagnating
            selected_state = None
            for s in sorted_states:
                if s['stagnation_count'] < stagnation_threshold:
                    selected_state = s
                    break
            
            if selected_state is None:
                # All are stagnating, pick the best
                selected_state = sorted_states[0]

            # Perform (1+1)-ES step with CSA
            x = selected_state['x']
            sigma = selected_state['sigma']
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            # Update CSA history and sigma
            # h_s = theta * h_s + (1 - theta) * (y - x) / sigma
            # Note: (y-x) is the step taken
            step = (y - x) / sigma
            selected_state['h_s'] = theta * selected_state['h_s'] + (1 - theta) * step
            
            # Update sigma
            # sigma = sigma * exp( c_s / d_s * ( ||h_s|| - chi_n ) )
            norm_h = np.linalg.norm(selected_state['h_s'])
            selected_state['sigma'] = sigma * np.exp((c_s / d_s) * (norm_h - chi_n))
            
            # Clamp sigma to reasonable bounds
            selected_state['sigma'] = np.clip(selected_state['sigma'], 1e-6, 10.0)

            # Check improvement
            if fy < selected_state['fx']:
                selected_state['x'] = y
                selected_state['fx'] = fy
                selected_state['stagnation_count'] = 0
                selected_state['last_improvement_call'] = calls_used
                
                # Update best overall
                if fy < best_overall_fx:
                    best_overall_fx = fy
                    best_overall_x = y.copy()
            else:
                selected_state['stagnation_count'] += 1

        return best_overall_x, best_overall_fx
# EVOLVE-BLOCK-END
