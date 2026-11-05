        new_energy = energy(current_indices, current_counts)
        delta_E = new_energy - old_energy
        
        # STRICT VALIDITY CHECK
        if np.any(current_counts == 0):
            # Move resulted in invalid covering, reject immediately
            undo_move(move)
            no_improve_count += 1
            T *= cooling_rate
            if T < T_end:
                T = T_end
            continue

        # Accept or reject based on energy (block count)
        if delta_E < 0 or (T > T_end and rng.random() < np.exp(-delta_E / T)):
            # Accept
            if new_energy < best_energy:
                best_energy = new_energy
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject - undo
            undo_move(move)
            no_improve_count += 1
