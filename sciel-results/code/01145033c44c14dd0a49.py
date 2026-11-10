<<<<<<< SEARCH
    def simulated_annealing(start_mask, rs, max_iters=2000, T0=1.0, cooling=0.99):
        """Simulated Annealing to escape local optima"""
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        T = T0
        
        # Precompute list of all vertices for random selection
        all_verts = list(range(n_rem))
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            # Pick a random vertex to flip (add or remove)
            v = rs.choice(all_verts)
            bit = 1 << v
            
            if current_mask & bit:
                # Try removing v
                new_mask = current_mask & ~bit
                new_size = current_size - 1
                delta = new_size - current_size # -1
            else:
                # Try adding v
                # Check conflicts
                conflicts = adj_mask[v] & current_mask
                if conflicts != 0:
                    # Cannot add directly. 
                    # For simplicity in SA, we only allow single-bit flips that result in valid states?
                    # No, adding v with conflicts is invalid.
                    # We must remove conflicts first. That's a multi-bit flip.
                    # Standard SA on independent set usually allows "move" operations:
                    # 1. Add v (if valid)
                    # 2. Remove v
                    # 3. Swap: Remove C, Add v (where C is set of conflicts)
                    
                    # Let's stick to valid moves only for simplicity, or allow "repair".
                    # If we add v with conflicts, the state is invalid.
                    # So we must ensure the new state is an independent set.
                    # Thus, we can only add v if conflicts is empty.
                    # Or we can define the move as "Remove C, Add v".
                    
                    # Let's try "Remove C, Add v"
                    new_mask = (current_mask & ~conflicts) | bit
                    new_size = current_size - bin(conflicts).count('1') + 1
                    delta = new_size - current_size
                else:
                    new_mask = current_mask | bit
                    new_size = current_size + 1
                    delta = 1
            
            # Acceptance criterion
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Accept worse with probability exp(delta / T)
                # delta is negative
                if rs.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        # Convert best_mask to list
        result = []
        temp = best_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result
=======
    def simulated_annealing(start_mask, rs, max_iters=2000, T0=1.0, cooling=0.99):
        """Simulated Annealing to escape local optima"""
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        T = T0
        
        # Precompute list of all vertices for random selection
        all_verts = list(range(n_rem))
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            # Pick a random vertex to flip (add or remove)
            v = rs.choice(all_verts)
            bit = 1 << v
            
            if current_mask & bit:
                # Try removing v
                new_mask = current_mask & ~bit
                new_size = current_size - 1
                delta = new_size - current_size # -1
            else:
                # Try adding v
                # Check conflicts
                conflicts = adj_mask[v] & current_mask
                if conflicts != 0:
                    # Try swap: remove one random conflict, add v
                    conflict_list = []
                    ct = conflicts
                    while ct:
                        lsb = ct & (-ct)
                        conflict_list.append(lsb.bit_length() - 1)
                        ct ^= lsb
                    cv = rs.choice(conflict_list)
                    cbit = 1 << cv
                    # Check validity
                    if adj_mask[v] & (current_mask & ~cbit) == 0:
                        new_mask = (current_mask & ~cbit) | bit
                        new_size = current_size  # swap, same size
                        delta = 0
                    else:
                        # Invalid swap, skip
                        continue
                else:
                    new_mask = current_mask | bit
                    new_size = current_size + 1
                    delta = 1
            
            # Acceptance criterion
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Accept worse with probability exp(delta / T)
                # delta is negative
                if rs.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        # Convert best_mask to list
        result = []
        temp = best_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result
>>>>>>> REPLACE

