        def __call__(self, func):
            dim = self.dim
            deadline = time.time() + 290
            
            # 1. Evaluate center
            x_best = np.zeros(dim)
            f_best = func(x_best)
            calls = 1
            
            # 2. Sequentially optimize each dimension
            for i in range(dim):
                if time.time() > deadline:
                    break
                
                # Probe positive bound
                x_probe_pos = x_best.copy()
                x_probe_pos[i] = 5.0
                f_probe_pos = func(x_probe_pos)
                calls += 1
                
                if f_probe_pos < f_best:
                    x_best = x_probe_pos
                    f_best = f_probe_pos
                else:
                    # Probe negative bound
                    if time.time() > deadline:
                        break
                    x_probe_neg = x_best.copy()
                    x_probe_neg[i] = -5.0
                    f_probe_neg = func(x_probe_neg)
                    calls += 1
                    
                    # Since f is linear, one of the bounds must be better than the center
                    # (or the current best which is at a bound).
                    # Actually, if f_probe_pos >= f_best, and x_best[i] was 0,
                    # then f_probe_neg < f_best.
                    # If x_best[i] was 5, we wouldn't be probing 5 again.
                    # We are probing the "other" side.
                    # So f_probe_neg should be < f_best.
                    if f_probe_neg < f_best:
                        x_best = x_probe_neg
                        f_best = f_probe_neg
            
            # 3. Fill remaining budget
            while calls < self.budget:
                if time.time() > deadline:
                    break
                func(x_best)
                calls += 1
                
            return x_best, f_best
        