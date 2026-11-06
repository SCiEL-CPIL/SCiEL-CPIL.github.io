    def sa_search(num_colors, time_limit):
        end_time = time.time() + time_limit
        best_colors = None
        best_violations = float('inf')
        
        while time.time() < end_time:
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]
            
            # Initial violation count
            # We can compute this efficiently
            current_violations = 0
            # Precompute AP positions as lists for fast access
            # ap_positions is already a list of lists
            
            # Let's just count all violations initially
            for ap in ap_positions:
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c0, c1, c2, c3}) < 3:
                    current_violations += 1
            
            if current_violations == 0:
                return True, colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
            
            temperature = 1.0
            cooling_rate = 0.9999
            
            while time.time() < end_time:
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    ap = ap_positions[ap_idx]
                    # Get colors
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    old_unique = len({c0, c1, c2, c3})
                    
                    # Update color in temporary check
                    # We don't want to modify colors array yet
                    # Just check the new unique count
                    # Create a list of colors for this AP
                    ap_colors = [colors[p] for p in ap]
                    # Find index of pos in ap
                    # Since ap is (a, a+d, a+2d, a+3d), we can check which one is pos
                    # Or just replace in the list
                    # ap.index(pos) is O(4)
                    idx_in_ap = ap.index(pos)
                    ap_colors[idx_in_ap] = new_color
                    new_unique = len(set(ap_colors))
                    
                    old_bad = 1 if old_unique < 3 else 0
                    new_bad = 1 if new_unique < 3 else 0
                    delta += (new_bad - old_bad)
                
                new_violations = current_violations + delta
                
                if new_violations < current_violations:
                    colors[pos] = new_color
                    current_violations = new_violations
                    if current_violations == 0:
                        return True, colors
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                else:
                    # Accept with prob exp(-delta/T)
                    if delta > 0:
                        if delta < 50:
                            if random.random() < np.exp(-delta / temperature):
                                colors[pos] = new_color
                                current_violations = new_violations
                        # else reject
                    else:
                        # delta == 0
                        if random.random() < 0.5:
                            colors[pos] = new_color
                            # current_violations unchanged
                
                temperature *= cooling_rate
                if temperature < 0.01:
                    break
        
        if best_colors is not None:
            # Final check
            v = 0
            for ap in ap_positions:
                c0, c1, c2, c3 = best_colors[ap[0]], best_colors[ap[1]], best_colors[ap[2]], best_colors[ap[3]]
                if len({c0, c1, c2, c3}) < 3:
                    v += 1
            if v == 0:
                return True, best_colors
            return False, best_colors
        return False, None
