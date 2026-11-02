<<<<<<< SEARCH
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Randomly pick one to diversify
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
=======
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Randomly pick one to diversify
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            # Additional randomization: with small probability, pick from top-3 gains instead
            if random.random() < 0.1:
                sorted_candidates = np.argsort(-gain)
                top_candidates = [c for c in sorted_candidates[:10] if not is_selected[c] and gain[c] >= max_gain - 2]
                if top_candidates:
                    b_idx = top_candidates[random.randint(0, len(top_candidates) - 1)]
>>>>>>> REPLACE

And increase restarts since the redundancy removal is now faster:

