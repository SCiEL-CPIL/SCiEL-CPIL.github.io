<<<<<<< SEARCH
    best_n = float('inf')
    best_edges = None
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        found = False
        max_attempts = 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            voltages = np.random.randint(1, m, size=15).astype(np.int64)
            
            # Local Repair Loop
            for repair_step in range(50):
                if time.time() > deadline:
                    break
                
                # Vectorized constraint checking
                # Compute sum(c_e * x_e) mod m for all constraints
                products = coeff_matrix @ voltages  # shape: (n_constraints,)
                residues = np.mod(products, m)
                
                # Find violated constraints (residue == 0)
                violated_indices = np.where(residues == 0)[0]
                
                if len(violated_indices) == 0:
                    # All constraints satisfied
                    voltages_list = voltages.tolist()
                    edges = _build_lifted_graph(pet_edges, voltages_list, m)
                    return n_candidate, edges
                
                # Pick a random violated constraint
                violated_idx = violated_indices[random.randint(0, len(violated_indices)-1)]
                violated_c = coeff_matrix[violated_idx]
                
                # Pick a random edge with non-zero coefficient in this constraint
                nonzero_indices = np.where(violated_c != 0)[0]
                if len(nonzero_indices) == 0:
                    break
                j = nonzero_indices[random.randint(0, len(nonzero_indices)-1)]
                
                # Change voltage j to a random new value
                voltages[j] = random.randint(1, m-1)
        
        # If we haven't found a solution for this m, try next m
    
    # Fallback: use LUW construction
    return _construct_luw(k=3, g=g, d=11)
=======
    best_n = float('inf')
    best_edges = None
    
    # Try both primes and some composites in increasing order
    m_candidates = []
    seen_m = set()
    for p in primes:
        if p > 200:
            break
        m_candidates.append(p)
        seen_m.add(p)
    # Add some composites that might work
    for cm in [6, 10, 12, 15, 21, 22, 26, 33, 34, 35, 38, 39, 46, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95]:
        if cm not in seen_m and cm > 2:
            m_candidates.append(cm)
            seen_m.add(cm)
    m_candidates.sort()
    
    for m in m_candidates:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            voltages = np.random.randint(1, m, size=15).astype(np.int64)
            
            # Local Repair Loop with targeted fixes
            for repair_step in range(100):
                if time.time() > deadline:
                    break
                
                # Vectorized constraint checking
                products = coeff_matrix @ voltages  # shape: (n_constraints,)
                residues = np.mod(products, m)
                
                # Find violated constraints (residue == 0)
                violated_indices = np.where(residues == 0)[0]
                
                if len(violated_indices) == 0:
                    # All constraints satisfied
                    voltages_list = voltages.tolist()
                    edges = _build_lifted_graph(pet_edges, voltages_list, m)
                    return n_candidate, edges
                
                # Try to fix violated constraints with targeted voltage changes
                fixed_any = False
                for vi in violated_indices[:5]:  # try up to 5 violated constraints
                    if time.time() > deadline:
                        break
                    violated_c = coeff_matrix[vi]
                    nonzero_indices = np.where(violated_c != 0)[0]
                    if len(nonzero_indices) == 0:
                        continue
                    
                    # For each edge in this constraint, try to find a voltage change
                    # that fixes this constraint
                    current_residue = residues[vi]
                    if current_residue == 0:
                        # Need to change sum by a non-zero amount mod m
                        for j in nonzero_indices:
                            c_j = violated_c[j]
                            old_v = voltages[j]
                            # We need new_v such that (old_sum - c_j*old_v + c_j*new_v) % m != 0
                            # i.e., c_j * (new_v - old_v) % m != 0
                            # Try a few candidate new values
                            for delta in [1, -1, 2, -2, 3, m//3, -m//3]:
                                new_v = (old_v + delta) % m
                                if new_v == 0:
                                    continue
                                # Check if this fixes the constraint
                                new_sum = int(products[vi]) - int(c_j) * int(old_v) + int(c_j) * int(new_v)
                                if new_sum % m != 0:
                                    # Check it doesn't break other constraints too badly
                                    # Quick check: just apply and see
                                    voltages[j] = new_v
                                    fixed_any = True
                                    break
                            if fixed_any:
                                break
                    if fixed_any:
                        break
                
                if not fixed_any:
                    # Fallback: random perturbation
                    j = random.randint(0, 14)
                    voltages[j] = random.randint(1, m-1)
        
        # If we haven't found a solution for this m, try next m
    
    # Fallback: use LUW construction
    return _construct_luw(k=3, g=g, d=11)
>>>>>>> REPLACE
