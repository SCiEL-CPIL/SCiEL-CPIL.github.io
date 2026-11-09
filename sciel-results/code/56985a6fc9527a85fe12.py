<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin for 120s limit
    
    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Initialize
        colours = np.array([(i + j) % K for i in range(N) for j in range(i + 1, N)], dtype=int)
        
        violations = count_violations(colours, k4_edge_ids)
        current_viol_count = np.sum(violations)
        
        if current_viol_count == 0:
            # Valid found
            return colours.tolist()
            
        # Simulated Annealing / Hill Climbing
        best_colours = colours.copy()
        best_viol_count = current_viol_count
        
        temp = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 2:
            if current_viol_count == 0:
                return colours.tolist()
                
            # Pick a random violating K4
            viol_indices = np.where(violations)[0]
            if len(viol_indices) == 0:
                break
            k4_id = viol_indices[random.randint(0, len(viol_indices) - 1)]
            e_ids = k4_edge_ids[k4_id]
            
            # Pick a random edge in this K4
            e_id = e_ids[random.randint(0, 5)]
            
            # Pick a new colour
            new_c = random.randint(0, K - 1)
            old_c = colours[e_id]
            
            if new_c == old_c:
                continue
                
            # Tentative change
            colours[e_id] = new_c
            
            # Update violations for K4s containing e_id
            affected_k4s = edge_to_k4s[e_id]
            new_viol_count = 0 # We only need to check if total goes down, but we maintain the array
            
            # To be safe and simple, we re-evaluate only the affected K4s and update the global count
            # But we need the full array for "pick random violating K4".
            # Optimization: Just update the bool array for affected k4s.
            
            delta = 0
            for ak4 in affected_k4s:
                old_val = violations[ak4]
                new_val = not check_k4(colours, k4_edge_ids[ak4])
                if old_val and not new_val:
                    delta -= 1
                elif not old_val and new_val:
                    delta += 1
                violations[ak4] = new_val
                
            new_total = current_viol_count + delta
            
            if new_total < current_viol_count:
                current_viol_count = new_total
                if current_viol_count < best_viol_count:
                    best_viol_count = current_viol_count
                    best_colours = colours.copy()
            else:
                # Accept with probability exp(-delta / temp)
                if delta > 0:
                    prob = np.exp(-delta / temp)
                    if random.random() < prob:
                        current_viol_count = new_total
                    else:
                        # Revert
                        colours[e_id] = old_c
                        for ak4 in affected_k4s:
                            # Revert violation state
                            violations[ak4] = not check_k4(colours, k4_edge_ids[ak4])
                else:
                    current_viol_count = new_total
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
                # Restart with new init if stuck? Or just keep going.
                
        if best_viol_count == 0:
            return best_colours.tolist()
            
    # Fallback: return the 17-colour solution if we couldn't find better
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
            
        # Multiple initializations
        best_overall_colours = None
        best_overall_viol = float('inf')
        
        for init_round in range(8):
            if time.time() > deadline - 5:
                break
                
            # Different initialization strategies
            if init_round == 0:
                # (i+j) % K
                colours = np.array([(i + j) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            elif init_round == 1:
                # (i*3 + j*5) % K
                colours = np.array([(i * 3 + j * 5) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            elif init_round == 2:
                # (i*7 + j*11) % K
                colours = np.array([(i * 7 + j * 11) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            elif init_round == 3:
                # Random
                colours = np.random.randint(0, K, 120).astype(np.int32)
            elif init_round == 4:
                # (i*j) % K
                colours = np.array([(i * j) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            elif init_round == 5:
                # (i + 2*j) % K
                colours = np.array([(i + 2 * j) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            elif init_round == 6:
                # (i*5 + j*2) % K
                colours = np.array([(i * 5 + j * 2) % K for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
            else:
                # Random with bias toward low values
                colours = np.random.randint(0, K, 120).astype(np.int32)
                colours[colours > K // 2] = np.random.randint(0, K // 2 + 1, (colours > K // 2).sum()).astype(np.int32)
            
            violations = compute_violations(colours)
            current_viol_count = int(np.sum(violations))
            
            if current_viol_count == 0:
                return colours.tolist()
            
            # Simulated Annealing
            best_colours = colours.copy()
            best_viol_count = current_viol_count
            
            temp = 2.0
            cooling = 0.9995
            min_temp = 0.05
            
            while time.time() < deadline - 3:
                if current_viol_count == 0:
                    return colours.tolist()
                
                # Pick a random violating K4
                viol_indices = np.where(violations)[0]
                if len(viol_indices) == 0:
                    break
                k4_id = viol_indices[random.randint(0, len(viol_indices) - 1)]
                e_ids = k4_edge_ids[k4_id]
                
                # With 30% probability, do a 2-edge move
                if random.random() < 0.3:
                    # Pick two distinct edges in this K4
                    e1_idx = random.randint(0, 5)
                    e2_idx = random.randint(0, 5)
                    while e2_idx == e1_idx:
                        e2_idx = random.randint(0, 5)
                    e1 = e_ids[e1_idx]
                    e2 = e_ids[e2_idx]
                    
                    new_c1 = random.randint(0, K - 1)
                    new_c2 = random.randint(0, K - 1)
                    old_c1 = colours[e1]
                    old_c2 = colours[e2]
                    
                    if new_c1 == old_c1 and new_c2 == old_c2:
                        continue
                    
                    colours[e1] = new_c1
                    colours[e2] = new_c2
                    
                    # Update violations for K4s containing e1 or e2
                    affected_k4s = set(edge_to_k4s[e1]) | set(edge_to_k4s[e2])
                    
                    delta = 0
                    for ak4 in affected_k4s:
                        old_val = violations[ak4]
                        new_val = not check_k4(colours, k4_edge_ids[ak4])
                        if old_val and not new_val:
                            delta -= 1
                        elif not old_val and new_val:
                            delta += 1
                        violations[ak4] = new_val
                    
                    new_total = current_viol_count + delta
                    
                    if new_total < current_viol_count:
                        current_viol_count = new_total
                        if current_viol_count < best_viol_count:
                            best_viol_count = current_viol_count
                            best_colours = colours.copy()
                    else:
                        if delta > 0:
                            prob = np.exp(-delta / temp)
                            if random.random() < prob:
                                current_viol_count = new_total
                            else:
                                colours[e1] = old_c1
                                colours[e2] = old_c2
                                for ak4 in affected_k4s:
                                    violations[ak4] = not check_k4(colours, k4_edge_ids[ak4])
                        else:
                            current_viol_count = new_total
                else:
                    # Single edge move
                    e_id = e_ids[random.randint(0, 5)]
                    
                    # Pick a new colour - bias toward colours not already used in this K4
                    k4_colours = set(colours[e] for e in e_ids)
                    available = [c for c in range(K) if c not in k4_colours]
                    if available and random.random() < 0.7:
                        new_c = random.choice(available)
                    else:
                        new_c = random.randint(0, K - 1)
                    
                    old_c = colours[e_id]
                    
                    if new_c == old_c:
                        continue
                    
                    colours[e_id] = new_c
                    
                    affected_k4s = edge_to_k4s[e_id]
                    
                    delta = 0
                    for ak4 in affected_k4s:
                        old_val = violations[ak4]
                        new_val = not check_k4(colours, k4_edge_ids[ak4])
                        if old_val and not new_val:
                            delta -= 1
                        elif not old_val and new_val:
                            delta += 1
                        violations[ak4] = new_val
                    
                    new_total = current_viol_count + delta
                    
                    if new_total < current_viol_count:
                        current_viol_count = new_total
                        if current_viol_count < best_viol_count:
                            best_viol_count = current_viol_count
                            best_colours = colours.copy()
                    else:
                        if delta > 0:
                            prob = np.exp(-delta / temp)
                            if random.random() < prob:
                                current_viol_count = new_total
                            else:
                                colours[e_id] = old_c
                                for ak4 in affected_k4s:
                                    violations[ak4] = not check_k4(colours, k4_edge_ids[ak4])
                        else:
                            current_viol_count = new_total
                
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
            
            if best_viol_count < best_overall_viol:
                best_overall_viol = best_viol_count
                best_overall_colours = best_colours.copy()
            
            if best_viol_count == 0:
                return best_colours.tolist()
        
        if best_overall_viol == 0:
            return best_overall_colours.tolist()
            
    # Fallback: return a valid 17-colour solution
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
>>>>>>> REPLACE
