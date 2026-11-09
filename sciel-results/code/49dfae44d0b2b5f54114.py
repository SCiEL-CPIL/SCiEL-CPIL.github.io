"""Multi-objective optimizer for WFG4 using LHS initialization + local refinement."""
import numpy as np
import time


class Optimizer:
    """LHS-based initialization with local Nelder-Mead refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Phase 1: Latin Hypercube Sampling for initial coverage
        # Use a reasonable number of initial points
        n_lhs = min(40, budget // 5)
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_X[:, d] = (perm + rng.random(n_lhs)) / n_lhs
        lhs_X = lo + lhs_X * (hi - lo)

        # Evaluate all LHS points
        F = np.array([problem(x) for x in lhs_X])
        used = n_lhs

        # Keep track of all evaluated points
        all_X = list(lhs_X)
        all_F = list(F)

        def check_time():
            return time.time() >= deadline

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            # f is dominated if there exists g in F_set such that g <= f and g != f
            for g in F_set:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        # Phase 2: Local refinement around non-dominated points
        # Identify non-dominated points from LHS
        non_dominated_idx = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
            if not dominated:
                non_dominated_idx.append(i)

        # Shuffle non-dominated points for diversity in refinement
        rng.shuffle(non_dominated_idx)

        # For each non-dominated point, do a few local Nelder-Mead-like steps
        # Budget remaining
        remaining = budget - used
        n_refine = len(non_dominated_idx)
        if n_refine > 0:
            evals_per_point = max(1, remaining // n_refine)
        else:
            evals_per_point = 0

        for idx in non_dominated_idx:
            if check_time():
                break
            if used >= budget:
                break

            x0 = lhs_X[idx].copy()
            f0 = F[idx].copy()

            # Simple local search: random perturbations with acceptance if non-dominated
            # Use a decreasing step size
            base_step = 0.15
            for step_i in range(evals_per_point):
                if check_time():
                    break
                if used >= budget:
                    break

                # Generate a few candidate perturbations
                n_candidates = 3
                best_candidate = None
                best_candidate_f = None

                for c in range(n_candidates):
                    if check_time():
                        break
                    if used >= budget:
                        break

                    step_size = base_step * (0.7 ** step_i)
                    perturbation = rng.standard_normal(dim) * step_size * (hi - lo)
                    candidate = np.clip(x0 + perturbation, lo, hi)
                    fc = problem(candidate)
                    used += 1

                    # Accept if it's non-dominated with respect to current best set
                    # Simple check: is fc better than f0 in at least one objective
                    # and not worse in all?
                    if np.any(fc < f0):
                        # Check if it dominates f0
                        if np.all(fc <= f0) and np.any(fc < f0):
                            best_candidate = candidate
                            best_candidate_f = fc
                            break
                        else:
                            # Keep track of best non-dominated candidate
                            if best_candidate_f is None:
                                best_candidate = candidate
                                best_candidate_f = fc
                            else:
                                # Prefer the one that dominates the other
                                if np.all(fc <= best_candidate_f) and np.any(fc < best_candidate_f):
                                    best_candidate = candidate
                                    best_candidate_f = fc

                if best_candidate is not None and best_candidate_f is not None:
                    x0 = best_candidate
                    f0 = best_candidate_f
                    all_X.append(x0.copy())
                    all_F.append(f0.copy())

        # Phase 3: If we have budget left, do more global random sampling
        if not check_time() and used < budget:
            while used < budget:
                if check_time():
                    break
                x_rand = lo + rng.random(dim) * (hi - lo)
                fr = problem(x_rand)
                used += 1
                all_X.append(x_rand)
                all_F.append(fr)

        # Return the best set of points (non-dominated from all evaluations)
        # For scoring, all recorded objective vectors are used, so we just return
        # something (the return value is ignored per the contract)
        return all_X[-1] if all_X else lo
