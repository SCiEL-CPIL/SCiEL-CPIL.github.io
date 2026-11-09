"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential-evolution-style multi-objective optimizer with archive-guided search and polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 325.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        rng = np.random

        # --- Initialization: mix structured and random points ---
        pop_size = min(40, max(15, budget // 5))

        # Generate corners (2^5 = 32)
        corners = []
        for i in range(2 ** dim):
            corner = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(corner)

        # Structured midpoints
        structured_extra = [
            (lo + hi) / 2.0,
            lo + 0.25 * span,
            lo + 0.75 * span,
            lo + 0.1 * span,
            lo + 0.9 * span,
        ]

        init_points = []
        # Add corners (sample if too many)
        n_corners = min(len(corners), pop_size // 2)
        if n_corners > 0:
            corner_idx = rng.choice(len(corners), size=n_corners, replace=False)
            init_points.extend([corners[i] for i in corner_idx])

        # Add structured extras
        for sp in structured_extra:
            if len(init_points) < pop_size:
                init_points.append(sp)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + span * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Archive management (vectorized for speed) ---
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            if not archive_F:
                archive_X.append(x.copy())
                archive_F.append(f.copy())
                return
            # Check if f is dominated by any archive point
            # Vectorized dominance check
            A = np.array(archive_F)
            # A[i] <= f and A[i] < f in at least one dim
            dominated = np.all(A <= f, axis=1) & np.any(A < f, axis=1)
            if np.any(dominated):
                return
            # Remove archive points dominated by f
            # f <= A[i] and f < A[i] in at least one dim
            dominates = np.all(f <= A, axis=1) & np.any(f < A, axis=1)
            keep = ~dominates
            archive_X = [archive_X[i] for i in range(len(archive_X)) if keep[i]]
            archive_F = [archive_F[i] for i in range(len(archive_F)) if keep[i]]
            archive_X.append(x.copy())
            archive_F.append(f.copy())

        for i in range(len(X)):
            update_archive(X[i], F[i])

        # --- Main DE loop with adaptive parameters ---
        # Start with higher exploration, shift to exploitation
        main_budget = budget - max(30, budget // 8)
        
        # Track progress for adaptive strategy
        best_sum = np.sum(np.min(F, axis=0))
        stagnation = 0
        last_best = best_sum

        while used < main_budget:
            if time.time() - start_time > DEADLINE:
                break

            # Adaptive parameters based on progress
            progress = used / max(1, main_budget)
            # CR: start high for exploration, decrease for exploitation
            CR = 0.7 - 0.3 * progress
            # F_mut: start moderate, decrease for finer search
            F_mut = 0.5 - 0.2 * progress

            # Select base: prefer archive if available and large enough
            if len(archive_X) >= 3 and rng.random() < 0.65:
                base_idx = rng.integers(len(archive_X))
                base_x = archive_X[base_idx]
                base_from_archive = True
            else:
                base_idx = rng.integers(pop_size)
                base_x = X[base_idx]
                base_from_archive = False

            # Select two other parents
            if len(archive_X) >= 3:
                other_idx = rng.choice(len(archive_X), size=2, replace=False)
                x1 = archive_X[other_idx[0]]
                x2 = archive_X[other_idx[1]]
            else:
                other_idx = rng.choice(pop_size, size=2, replace=False)
                x1 = X[other_idx[0]]
                x2 = X[other_idx[1]]

            # Mutation
            mutant = base_x + F_mut * (x1 - x2)
            mutant = np.clip(mutant, lo, hi)

            # Binomial crossover
            trial = base_x.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = np.array(problem(trial))
            used += 1

            # Update archive
            update_archive(trial, f_trial)

            # Track best for stagnation detection
            current_best = np.sum(np.min(F, axis=0))
            if current_best < last_best - 1e-6:
                last_best = current_best
                stagnation = 0
            else:
                stagnation += 1

            # Population replacement
            if base_from_archive:
                # Don't replace population when using archive as base
                pass
            else:
                base_f = F[base_idx]
                # Replace if trial dominates base
                if np.all(f_trial <= base_f) and np.any(f_trial < base_f):
                    X[base_idx] = trial
                    F[base_idx] = f_trial
                # If non-dominated, replace with some probability or if better sum
                elif not (np.all(base_f <= f_trial) and np.any(base_f < f_trial)):
                    # Non-dominated: use a simple criterion
                    if np.sum(f_trial) < np.sum(base_f) or rng.random() < 0.4:
                        X[base_idx] = trial
                        F[base_idx] = f_trial

            # Periodic diversity injection - more frequent when stagnant
            inject_freq = max(10, 20 - stagnation // 2)
            if used % inject_freq == 0 and used < main_budget:
                if time.time() - start_time > DEADLINE:
                    break
                # Mix of random and archive-neighborhood injection
                if rng.random() < 0.5 and archive_X:
                    # Perturb an archive point
                    base_arch = rng.integers(len(archive_X))
                    rand_x = archive_X[base_arch] + rng.normal(0, 0.3 * span, dim)
                else:
                    rand_x = lo + span * rng.random(dim)
                rand_x = np.clip(rand_x, lo, hi)
                rand_f = np.array(problem(rand_x))
                used += 1
                update_archive(rand_x, rand_f)
                # Replace worst in population by sum
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = rand_x
                F[worst_idx] = rand_f

        # --- Polish phase: more aggressive local search ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

            # Sort archive by sum of objectives, take top few
            sums = np.array([np.sum(af) for af in archive_F])
            top_n = min(15, len(archive_X))
            top_idx = np.argsort(sums)[:top_n]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Multi-scale local search: try different sigma values
                sigmas = [0.15, 0.08, 0.04, 0.02, 0.01]
                for sigma_scale in sigmas:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break

                    sigma = sigma_scale * span
                    # Try 3 perturbations at this scale
                    for _ in range(3):
                        if used >= budget:
                            break
                        if time.time() - start_time > DEADLINE:
                            break
                        perturbed = center + rng.normal(0, sigma, dim)
                        perturbed = np.clip(perturbed, lo, hi)

                        f_pert = np.array(problem(perturbed))
                        used += 1

                        # Accept if it improves the center in a meaningful way
                        # Check if perturbed dominates center
                        if np.all(f_pert <= center_f) and np.any(f_pert < center_f):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert
                        # Or if non-dominated and closer to ideal point
                        elif not (np.all(center_f <= f_pert) and np.any(center_f < f_pert)):
                            # Use a simple improvement criterion
                            if np.sum(f_pert) < np.sum(center_f) * 0.95:
                                update_archive(perturbed, f_pert)
                                center = perturbed
                                center_f = f_pert

        return X
# EVOLVE-BLOCK-END
