<<<<<<< SEARCH
        def hooke_jees(start_x):
            nonlocal eval_count
            if eval_count >= budget - 10:  # Leave some buffer
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            step_sizes = np.ones(d) * 0.5
            min_step = 1e-4
            max_iter = 200  # Limit iterations to prevent infinite loops if stuck

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                        # Continue exploring this direction if it improved
                        # (Standard HJ often resets the loop or continues, 
                        # here we continue to the next coordinate to maintain 
                        # a structured search, but we could also loop back.
                        # Sticking to the inspiration's flow: continue to next i)
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step) or not moved:
                    # If no move was made in a full cycle, we might be stuck or converged
                    # If step sizes are small, we are converged.
                    if np.all(step_sizes < min_step):
                        break
                    # If no move but steps are large, try shrinking all steps slightly?
                    # Or just break if we didn't move at all to avoid wasting budget
                    if not moved:
                        break
=======
        def line_minimize(x, direction, f_current):
            """
            1D minimization along a line.
            Returns new_x, new_f.
            """
            nonlocal eval_count
            step = 0.1
            x_best = x.copy()
            f_best = f_current
            
            # Try to bracket the minimum
            # We evaluate x + step * dir and x - step * dir
            x_plus = np.clip(x + step * direction, -5, 5)
            x_minus = np.clip(x - step * direction, -5, 5)
            
            f_plus = eval_func(x_plus)
            f_minus = eval_func(x_minus)
            
            if eval_count >= budget:
                return x_best, f_best
            
            if f_plus < f_best and f_plus < f_minus:
                x_best = x_plus
                f_best = f_plus
                # Expand
                for _ in range(10):
                    if eval_count >= budget: break
                    x_new = np.clip(x_best + step * direction, -5, 5)
                    f_new = eval_func(x_new)
                    if f_new < f_best:
                        x_best = x_new
                        f_best = f_new
                        step *= 2.0
                    else:
                        break
            elif f_minus < f_best and f_minus < f_plus:
                x_best = x_minus
                f_best = f_minus
                # Expand
                for _ in range(10):
                    if eval_count >= budget: break
                    x_new = np.clip(x_best - step * direction, -5, 5)
                    f_new = eval_func(x_new)
                    if f_new < f_best:
                        x_best = x_new
                        f_best = f_new
                        step *= 2.0
                    else:
                        break
            else:
                # No improvement, shrink step and try again? 
                # Or just return current.
                # To avoid infinite loops, we just return.
                pass
                
            return x_best, f_best

        def powell(start_x):
            nonlocal eval_count
            if eval_count >= budget - 20:
                return

            x = np.clip(start_x, -5, 5).copy()
            f = eval_func(x)
            if eval_count >= budget:
                return

            directions = [np.eye(d)[i] for i in range(d)]
            max_iter = 50
            tol = 1e-8
            
            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x_start = x.copy()
                f_start = f
                min_delta = float('inf')
                min_i = -1
                max_delta = -float('inf')
                max_i = -1
                f_vals = []
                
                for i in range(d):
                    if eval_count >= budget:
                        break
                    x, f = line_minimize(x, directions[i], f)
                    f_vals.append(f)
                    delta = f_start - f
                    if delta > max_delta:
                        max_delta = delta
                        max_i = i
                    if delta < min_delta:
                        min_delta = delta
                        min_i = i

                if min_i == -1:
                    break

                x_new = x.copy()
                f_new = f
                
                # New direction
                new_dir = x_new - x_start
                if np.linalg.norm(new_dir) < 1e-5:
                    break
                
                # Minimize along new direction
                x_check, f_check = line_minimize(x_new, new_dir, f_new)
                
                # Powell's condition
                # If f_check < f_start - f_min (where f_min is min of f_vals)
                # Actually, standard condition:
                # If f_check < f_start - min(f_vals) + ...
                # Let's use: if f_check < f_start - min_delta
                # Note: min_delta is f_start - max(f_vals) (since delta = f_start - f)
                # So f_start - min_delta = max(f_vals)
                # This condition is: f_check < max(f_vals)
                # This means the new direction gave a better result than the worst direction in the previous cycle.
                
                if f_check < f_start - min_delta:
                    # Replace the direction with the least improvement (min_i) with new_dir
                    # And set x to x_check
                    directions[min_i] = new_dir / np.linalg.norm(new_dir)
                    x = x_check
                    f = f_check
                else:
                    # Replace the direction with the least improvement with new_dir
                    # But keep x as x_new (the result of the cycle)
                    directions[min_i] = new_dir / np.linalg.norm(new_dir)
                    x = x_new
                    f = f_new

                if max_delta < tol:
                    break

            return
>>>>>>> REPLACE

