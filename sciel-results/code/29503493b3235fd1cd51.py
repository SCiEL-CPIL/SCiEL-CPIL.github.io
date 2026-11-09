        def construct():
            t0 = time.time()
            DEADLINE = t0 + 140.0  # Safety margin

            # Initial guess: Hex lattice
            init_pts = _hex_lattice_init(1.0)
            current_params = np.zeros(2 * N)
            for i in range(N):
                current_params[2 * i] = init_pts[i, 0]
                current_params[2 * i + 1] = init_pts[i, 1]

            best_S = 10.0  # Start with a large, definitely valid S
            best_params = current_params.copy()
            center = (0.0, 0.0)
            outer_angle = 0.0

            # First, ensure we have a valid configuration for a large S
            # Try S=6.0
            S_test = 6.0
            if time.time() < DEADLINE - 10:
                refined = _refine(current_params, S_test, center, outer_angle, DEADLINE, max_iter=100)
                if _check_valid(refined, S_test, center, outer_angle, tol=1e-5):
                    best_S = S_test
                    best_params = refined
                    current_params = refined

            # Try to reduce S
            S_values = [5.0, 4.5, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5]

            for S in S_values:
                if S >= best_S:
                    continue
                if time.time() > DEADLINE - 5:
                    break

                # Refine positions for this S, using previous best as start
                refined = _refine(current_params, S, center, outer_angle, DEADLINE, max_iter=150)

                # Check validity
                if _check_valid(refined, S, center, outer_angle, tol=1e-6):
                    best_S = S
                    best_params = refined
                    current_params = refined
                else:
                    # If we can't fit in this S, we likely can't fit in smaller ones
                    # without a major reconfiguration which L-BFGS from current pos won't do.
                    # So we break.
                    break

            # Final polish with the best S found
            if best_S < 10.0 and time.time() < DEADLINE - 2:
                final = _refine(best_params, best_S, center, outer_angle, DEADLINE, max_iter=300)
                if _check_valid(final, best_S, center, outer_angle, tol=1e-7):
                    best_params = final

            inner = [[float(best_params[2 * i]), float(best_params[2 * i + 1]), 0.0] for i in range(N)]
            return inner, center, float(best_S), float(outer_angle)
        