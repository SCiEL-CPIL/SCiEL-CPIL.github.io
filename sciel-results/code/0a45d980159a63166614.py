    def try_recolor_and_add_v2(ei):
        # Get all active cycles for ei
        active_cycles = []
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                active_cycles.append(ci)
        
        if not active_cycles:
            return False
            
        random.shuffle(active_cycles)
        
        # Try to fix each conflict
        for ci in active_cycles[:5]: # Limit to 5 conflicts to try
            cyc = all_4cycles[ci]
            other_edges = [e2 for e2 in cyc if e2 != ei]
            random.shuffle(other_edges)
            
            for e_change in other_edges:
                old_color = colors[e_change]
                valid_colors = [c for c in range(n_colors) if c != old_color]
                random.shuffle(valid_colors)
                
                for new_c in valid_colors:
                    # Check validity for e_change
                    valid = True
                    for ci2 in edge_cycles[e_change]:
                        cyc2 = all_4cycles[ci2]
                        if all(e2 in edges_set for e2 in cyc2):
                            cs = [colors.get(e2) for e2 in cyc2 if e2 != e_change]
                            if new_c in cs:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    colors[e_change] = new_c
                    if try_add(ei):
                        return True
                    else:
                        colors[e_change] = old_color
                        
        return False
