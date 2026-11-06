            durations = list(machine_options.values())
            if not durations:
                min_duration = 1.0
                soft_min_duration = 1.0
            else:
                min_duration = min(durations)
                # Use a temperature that makes soft_min close to min but smoother
                # If durations are integers, T=1.0 is reasonable.
                soft_min_duration = -1.0 * np.logsumexp(-np.array(durations, dtype=float) / 1.0)
                
            if min_duration <= 0: min_duration = 1.0
            if soft_min_duration <= 0: soft_min_duration = 1.0
            if avg_duration <= 0: avg_duration = 1.0
            
            f1 = remaining_work
            f2 = 1.0 / soft_min_duration  # Use soft-min for SPT feature
            f3 = 1.0 / avg_duration
            f4 = 1.0 / remaining_ops if remaining_ops > 0 else 1.0
            f5 = f1 * f2 # Work / Soft-Min-Duration
            
            w1 = 1.0
            w2 = 2.0
            w3 = 0.5
            w4 = 0.5
            w5 = 4.0
            
            score = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4 + w5 * f5
