            if max_mult < current_peak:
                current_peak = max_mult
                stagnation = 0
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)
            elif max_mult == current_peak:
                # Accept with probability 0.2 to explore
                if random.random() < 0.2:
                    stagnation = 0
                    # Keep the new element (don't revert)
                    # Note: current_A and current_A_set already updated
                    if current_peak < best_peak:
                         best_peak = current_peak
                         best_A = list(current_A)
                else:
                    # Revert
                    current_A[idx] = old_elem
                    current_A_set.remove(new_elem)
                    current_A_set.add(old_elem)
                    stagnation += 1
            else:
                # Revert
                current_A[idx] = old_elem
                current_A_set.remove(new_elem)
                current_A_set.add(old_elem)
                stagnation += 1
