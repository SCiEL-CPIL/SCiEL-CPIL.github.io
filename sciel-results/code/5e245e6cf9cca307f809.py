    def reduce_solution(solution):
        if not solution: return []
        
        depth = np.zeros(n_t, dtype=np.int32)
        for bi in solution:
            for idx in block_indices[bi]:
                depth[idx] += 1
            
        new_sol = []
        for bi in solution:
            idxs = block_indices[bi]
            # Tentatively remove
            for idx in idxs:
                depth[idx] -= 1
            
            # Check min
            is_essential = False
            for idx in idxs:
                if depth[idx] == 0:
                    is_essential = True
                    break
            
            if is_essential:
                # Add back
                for idx in idxs:
                    depth[idx] += 1
                new_sol.append(bi)
            else:
                # Redundant
                pass
                
        if len(new_sol) < len(solution):
            return reduce_solution(new_sol)
        return new_sol
