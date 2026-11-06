            # Limit candidate positions to speed up
            if len(candidate_positions) > 20:
                candidates = random.sample(list(candidate_positions), 20)
            else:
                candidates = list(candidate_positions)
