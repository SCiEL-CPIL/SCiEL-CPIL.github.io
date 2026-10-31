<<<<<<< SEARCH
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        # Try multiple restarts
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            # Generate hexagonal initial layout
            # Estimate grid size
            # For N=21, 5 cols x 4 rows = 20, plus 1.
            # Let's try to fit N circles in a hexagonal grid.
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            
            # Adjust to ensure we have enough space
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs = np.zeros(N)
            ys = np.zeros(N)
            
            # Hexagonal packing coordinates
            # Horizontal spacing: 2*r
            # Vertical spacing: sqrt(3)*r
            # We don't know r yet, so we estimate based on grid density.
            # Assume r ~ min(w/n_cols, h/n_rows) / 1.1
            est_r = min(w / n_cols, h / n_rows) * 0.45
            
            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                
                # Center of cell
                x_center = (col + 0.5) * w / n_cols
                y_center = (row + 0.5) * h / n_rows
                
                # Stagger odd rows
                if row % 2 == 1:
                    x_center += 0.5 * w / n_cols
                    # Wrap around or clamp?
                    # If it goes out of bounds, we might need to adjust the grid.
                    # For simplicity, let's just shift and let the optimizer fix it,
                    # or clamp to stay inside.
                    if x_center + est_r > w:
                        x_center = w - est_r
                    elif x_center - est_r < 0:
                        x_center = est_r

                xs[i] = x_center
                ys[i] = y_center

            # Add random jitter to break symmetry
            if restart > 0:
                # Jitter amount should be small, relative to cell size
                jitter_scale = 0.1 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                
                # Ensure initial points are inside rectangle (roughly)
                # We'll let the optimizer handle fine tuning, but start inside
                # to help SLSQP.
                # Clip to [r0, w-r0] etc? We don't have r0 yet.
                # Let's just clip to [0.01, w-0.01]
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

            # Initial radii
            # Calculate min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)

            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            try:
                # Use SLSQP
                # maxiter: allow enough iterations to converge
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                # Check if result is valid and better
                # Even if result.success is False, it might be a good candidate
                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])

            except Exception:
                continue
=======
    def polish_result(x, y, r, w, h):
        """
        Local optimization pass to refine the packing.
        Maximizes sum of radii with stricter tolerances.
        """
        constraints = make_constraints(w, h)
        
        # Pack variables
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = x[i]
            x0[3*i+1] = y[i]
            x0[3*i+2] = r[i]
            
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            x_pol, y_pol, r_pol = unpack(result.x)
            
            if validate(x_pol, y_pol, r_pol, w, h):
                return x_pol, y_pol, r_pol, np.sum(r_pol)
        except Exception:
            pass
            
        return x, y, r, np.sum(r)

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        # Try multiple restarts
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            # Generate hexagonal initial layout
            # Estimate grid size
            # For N=21, 5 cols x 4 rows = 20, plus 1.
            # Let's try to fit N circles in a hexagonal grid.
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            
            # Adjust to ensure we have enough space
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs = np.zeros(N)
            ys = np.zeros(N)
            
            # Hexagonal packing coordinates
            # Horizontal spacing: 2*r
            # Vertical spacing: sqrt(3)*r
            # We don't know r yet, so we estimate based on grid density.
            # Assume r ~ min(w/n_cols, h/n_rows) / 1.1
            est_r = min(w / n_cols, h / n_rows) * 0.45
            
            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                
                # Center of cell
                x_center = (col + 0.5) * w / n_cols
                y_center = (row + 0.5) * h / n_rows
                
                # Stagger odd rows
                if row % 2 == 1:
                    x_center += 0.5 * w / n_cols
                    # Wrap around or clamp?
                    # If it goes out of bounds, we might need to adjust the grid.
                    # For simplicity, let's just shift and let the optimizer fix it,
                    # or clamp to stay inside.
                    if x_center + est_r > w:
                        x_center = w - est_r
                    elif x_center - est_r < 0:
                        x_center = est_r

                xs[i] = x_center
                ys[i] = y_center

            # Add random jitter to break symmetry
            if restart > 0:
                # Jitter amount should be small, relative to cell size
                jitter_scale = 0.1 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                
                # Ensure initial points are inside rectangle (roughly)
                # We'll let the optimizer handle fine tuning, but start inside
                # to help SLSQP.
                # Clip to [r0, w-r0] etc? We don't have r0 yet.
                # Let's just clip to [0.01, w-0.01]
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

            # Initial radii
            # Calculate min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)

            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            try:
                # Use SLSQP
                # maxiter: allow enough iterations to converge
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                # Check if result is valid and better
                # Even if result.success is False, it might be a good candidate
                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    # Polish the result
                    x_pol, y_pol, r_pol, score_pol = polish_result(x, y, r, w, h)
                    
                    if score_pol > best_score:
                        best_score = score_pol
                        best_result = np.column_stack([x_pol, y_pol, r_pol])

            except Exception:
                continue
>>>>>>> REPLACE

