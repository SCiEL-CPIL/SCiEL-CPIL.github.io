def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_colors = None
    best_distinct = 0
    
    # Run multiple short sessions to explore different regions
    while time.time() < deadline:
        colors = _construct_initial()
        current_distinct = len(set(colors))
        
        edges = list(range(NUM_EDGES))
        no_improve_counter = 0
        
        while time.time() < deadline:
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            while e2 == e1:
                e2 = random.choice(edges)
                
            c1 = random.randint(0, 119)
            c2 = random.randint(0, 119)
            
            old_c1 = colors[e1]
            old_c2 = colors[e2]
            
            colors[e1] = c1
            colors[e2] = c2
            
            valid = True
            # Check cycles
            # Optimization: only check cycles involving e1 or e2
            # Using precomputed lists
            for c_idx in _EDGE_CYCLES[e1]:
                cycle = _CYCLES[c_idx]
                c0, c1v, c2v, c3v = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                if c0 != c1v and c0 != c2v and c0 != c3v and c1v != c2v and c1v != c3v and c2v != c3v:
                    valid = False
                    break
            if valid:
                for c_idx in _EDGE_CYCLES[e2]:
                    cycle = _CYCLES[c_idx]
                    c0, c1v, c2v, c3v = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
                    if c0 != c1v and c0 != c2v and c0 != c3v and c1v != c2v and c1v != c3v and c2v != c3v:
                        valid = False
                        break
            
            if valid:
                new_distinct = len(set(colors))
                if new_distinct > current_distinct:
                    current_distinct = new_distinct
                    no_improve_counter = 0
                    if current_distinct > best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors[:]
                elif new_distinct < current_distinct:
                    colors[e1] = old_c1
                    colors[e2] = old_c2
                    no_improve_counter += 1
                else:
                    # Equal distinct count, keep move
                    no_improve_counter += 1
            else:
                colors[e1] = old_c1
                colors[e2] = old_c2
                no_improve_counter += 1
                
            # If stuck for too long, break and restart
            if no_improve_counter > 5000:
                break
                
    return best_colors if best_colors else [0]*NUM_EDGES
