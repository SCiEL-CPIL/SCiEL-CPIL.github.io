        # Local search for voltages
        best_voltages = None
        best_violations = float('inf')
        
        for restart in range(50):
            voltages = [rng.randint(0, p-1) for _ in range(num_edges)]
            
            # Calculate initial violations
            def count_violations(v):
                count = 0
                for L in range(5, g):
                    for w in constraints_by_len[L]:
                        s = 0
                        for e_idx, sign in w:
                            s += sign * v[e_idx]
                        if s % p == 0:
                            count += 1
                return count

            curr_violations = count_violations(voltages)
            
            if curr_violations == 0:
                best_voltages = voltages
                break
                
            # Local search
            for step in range(1000):
                if curr_violations == 0:
                    break
                # Pick a random edge to change
                e_idx = rng.randint(0, num_edges - 1)
                old_val = voltages[e_idx]
                new_val = rng.randint(0, p - 1)
                if new_val == old_val:
                    continue
                voltages[e_idx] = new_val
                new_violations = count_violations(voltages)
                if new_violations < curr_violations:
                    curr_violations = new_violations
                else:
                    voltages[e_idx] = old_val
                
            if curr_violations < best_violations:
                best_violations = curr_violations
                best_voltages = voltages[:]
                
        if best_violations == 0:
            found = True
            voltages = best_voltages
