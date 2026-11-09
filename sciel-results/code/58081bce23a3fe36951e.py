    def solve_n(n, time_limit):
        if n < 4:
            triples = list(combinations(range(n), 3))
            return [0] * len(triples)
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        triple_to_k4 = [[] for _ in range(num_triples)]
        triple_to_k5 = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4s):
            for tri in combinations(k4, 3):
                triple_to_k4[triple_idx[tri]].append(i)
                
        for i, k5 in enumerate(k5s):
            for tri in combinations(k5, 3):
                triple_to_k5[triple_idx[tri]].append(i)
                
        local_deadline = time.time() + time_limit
        best_violations = float('inf')
        best_colors = None
        
        # Multiple restarts
        while time.time() < local_deadline:
            colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            red_count_k4 = np.zeros(num_k4, dtype=np.int32)
            blue_count_k5 = np.zeros(num_k5, dtype=np.int32)
            
            for t in range(num_triples):
                if colors[t] == 0:
                    for k4_idx in triple_to_k4[t]:
                        red_count_k4[k4_idx] += 1
                else:
                    for k5_idx in triple_to_k5[t]:
                        blue_count_k5[k5_idx] += 1
                        
            red_violations = np.sum(red_count_k4 == 4)
            blue_violations = np.sum(blue_count_k5 == 10)
            total_violations = red_violations + blue_violations
            
            if total_violations == 0:
                return colors.tolist()
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            T = 10.0
            cooling_rate = 0.99
            # Run SA for a short burst or until time runs out
            # We'll just run the SA loop until the global deadline or a local stagnation
            sa_deadline = min(local_deadline, time.time() + 5.0) # 5s per restart
            
            while time.time() < sa_deadline:
                t_idx = random.randint(0, num_triples - 1)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                delta_red = 0
                delta_blue = 0
                
                if old_color == 0:
                    for k4_idx in triple_to_k4[t_idx]:
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 4: delta_red -= 1
                        elif old_cnt == 3: delta_red += 1
                        red_count_k4[k4_idx] = new_cnt
                else:
                    for k4_idx in triple_to_k4[t_idx]:
                        old_cnt = red_count_k4[k4_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 3: delta_red += 1
                        elif old_cnt == 4: delta_red -= 1
                        red_count_k4[k4_idx] = new_cnt
                        
                if old_color == 1:
                    for k5_idx in triple_to_k5[t_idx]:
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt - 1
                        if old_cnt == 10: delta_blue -= 1
                        elif old_cnt == 9: delta_blue += 1
                        blue_count_k5[k5_idx] = new_cnt
                else:
                    for k5_idx in triple_to_k5[t_idx]:
                        old_cnt = blue_count_k5[k5_idx]
                        new_cnt = old_cnt + 1
                        if old_cnt == 9: delta_blue += 1
                        elif old_cnt == 10: delta_blue -= 1
                        blue_count_k5[k5_idx] = new_cnt
                        
                delta_E = delta_red + delta_blue
                
                if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
                    colors[t_idx] = new_color
                    red_violations += delta_red
                    blue_violations += delta_blue
                    total_violations = red_violations + blue_violations
                    
                    if total_violations < best_violations:
                        best_violations = total_violations
                        best_colors = colors.copy()
                        if total_violations == 0:
                            return colors.tolist()
                else:
                    if old_color == 0:
                        for k4_idx in triple_to_k4[t_idx]:
                            red_count_k4[k4_idx] += 1
                    else:
                        for k4_idx in triple_to_k4[t_idx]:
                            red_count_k4[k4_idx] -= 1
                            
                    if old_color == 1:
                        for k5_idx in triple_to_k5[t_idx]:
                            blue_count_k5[k5_idx] += 1
                    else:
                        for k5_idx in triple_to_k5[t_idx]:
                            blue_count_k5[k5_idx] -= 1
                            
                T *= cooling_rate
                if T < 0.1:
                    T = 5.0

        return None
