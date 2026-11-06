        def backtrack_optimized():
            # Find best variable
            best_v = -1
            best_score = -1
            for v in range(n):
                if state[v] == 0:
                    # Count unassigned in incident edges
                    score = 0
                    for e_idx in incident[v]:
                        mask = edge_masks[e_idx]
                        for u in range(n):
                            if u != v and (mask & (1 << u)) and state[u] == 0:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                        
            if best_v == -1:
                return False # SAT
            
            saved_state = state[:] # Copy state
            
            # Try 0
            state[best_v] = 1
            if propagate():
                if backtrack_optimized():
                    return True
            
            # Restore state
            state = saved_state[:]
            
            # Try 1
            state[best_v] = 2
            if propagate():
                if backtrack_optimized():
                    return True
            
            # Restore state (not strictly necessary if returning False, but good practice)
            # Actually, if both fail, we return False (UNSAT for this branch)
            # The parent will restore its own state.
            
            return False # UNSAT
