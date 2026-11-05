            max_gain = gains.max()
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1 and noise_scale > 0:
                best_idx = np.random.choice(candidates)
            else:
                best_idx = candidates[0]
