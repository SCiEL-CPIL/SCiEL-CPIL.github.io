    def expand(idx, candidates, current_mask, current_count):
        # idx is the index in the 'order' list
        if idx == N:
             update best
             return
             
        if current_count + candidates.bit_count() <= best_size:
             return
             
        v = order[idx]
        
        if candidates & (1 << v):
            # Option 1: Include v
            new_c1 = candidates & ~(adj[v] | (1 << v))
            expand(idx + 1, new_c1, current_mask | (1 << v), current_count + 1)
            
            # Option 2: Exclude v
            new_c2 = candidates & ~(1 << v)
            expand(idx + 1, new_c2, current_mask, current_count)
        else:
            # v not in candidates, just move to next
            expand(idx + 1, candidates, current_mask, current_count)
    