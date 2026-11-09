            if violations < best_violations:
                best_violations = violations
                best_mask = red_mask
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > 500:
                    # Restart
                    red_mask = 0
                    for i in range(m):
                        if random.random() < 0.5:
                            red_mask |= (1 << i)
                    steps_since_improvement = 0
                    continue
