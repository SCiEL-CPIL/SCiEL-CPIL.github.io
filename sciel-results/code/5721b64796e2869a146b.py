    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Inside loop:
        # ...
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            # Update pos_to_idx
            # Remove old positions of k..n-1
            for i in range(k, n):
                del pos_to_idx[chain[i]] # Wait, chain is updated?
                # No, chain variable is updated to new_chain.
                # We need to delete OLD positions.
            
            # So we must save old suffix positions before updating chain.
