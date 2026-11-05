            def check_generators(gens, mult_table, index_map, g):
                dist = [-1] * 720
                dist[0] = 0
                queue = [0]
                head = 0
                min_cycle = float('inf')
                
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    du = dist[u]
                    
                    if du >= g:
                        continue # No need to expand further for girth < g? 
                        # Actually, if du >= g, any cycle found via u will be >= g+1?
                        # Cycle length = dist[u] + dist[v] + 1.
                        # If dist[u] >= g, then cycle >= g+1.
                        # So we only care about nodes with dist < g.
                        
                    for gi in gens:
                        v = mult_table[u][gi]
                        if dist[v] == -1:
                            dist[v] = du + 1
                            if dist[v] < g:
                                queue.append(v)
                        else:
                            # v is visited
                            # Cycle length
                            cycle_len = du + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                if min_cycle < g:
                                    return False
                            
                return True
            