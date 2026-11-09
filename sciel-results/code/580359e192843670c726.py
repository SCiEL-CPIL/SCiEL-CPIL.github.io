    def local_search_batch(residues, max_time):
        current = residues.copy()
        current_score = evaluate(current)
        
        if current_score >= 1000:
            return current, current_score

        T = 1.0
        cooling = 0.9999
        batch_size = 5
        
        while time.time() < max_time:
            # Pick a random batch of indices
            batch_indices = np.random.choice(n, size=batch_size, replace=False)
            
            # Generate new residues for these indices
            new_residues = current.copy()
            for idx in batch_indices:
                p = P_list[idx]
                new_residues[idx] = random.randint(0, p - 1)
                
            new_score = evaluate(new_residues)
            delta = new_score - current_score
            
            if delta > 0:
                current = new_residues
                current_score = new_score
                if current_score >= 1000:
                    return current, current_score
            else:
                # Accept with probability exp(delta / T)
                # delta is negative
                if random.random() < math.exp(delta / T):
                    current = new_residues
                    current_score = new_score
                    
            T *= cooling
            if T < 0.01:
                T = 0.01

        return current, current_score
