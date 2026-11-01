        elif rng.random() < 0.5:
            # Remove a random selected block
            selected_indices = [i for i in range(n_blocks) if selected[i]]
            if not selected_indices:
                continue
            bi = rng.choice(selected_indices)
            action = 'remove'
        else:
            # Add a random unselected block
            unselected_indices = [i for i in range(n_blocks) if not selected[i]]
            if not unselected_indices:
                continue
            bi = rng.choice(unselected_indices)
            action = 'add'
