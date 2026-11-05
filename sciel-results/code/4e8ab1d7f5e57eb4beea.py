<<<<<<< SEARCH
    def exchange_move():
        """Try to remove one codeword and add another to improve coverage efficiency."""
        nonlocal code, coverage
        if len(code) < 2:
            return False
            
        old_size = len(code)
        
        # Pick a codeword to remove
        c_remove = random.choice(code)
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        
        if not uncovered_set:
            # Removal is free
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True
        
        # Try to find a single replacement that covers all uncovered
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbor_sets[w])
        
        best_c = -1
        best_count = 0
        for c in candidate_pool:
            if c in code:
                continue
            cnt = sum(1 for x in neighbors_list[c] if x in uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_set):
                break
        
        if best_c != -1 and best_count == len(uncovered_set):
            # Single replacement works
            code.remove(c_remove)
            code.append(best_c)
            coverage[neighbors_list[c_remove]] -= 1
            coverage[neighbors_list[best_c]] += 1
            return True
        
        # Try greedy repair with multiple additions
        added = greedy_repair(uncovered_set)
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        
        return False
=======
    def exchange_move():
        """Try to remove one codeword and add another to improve coverage efficiency."""
        nonlocal code, coverage
        if len(code) < 2:
            return False
            
        old_size = len(code)
        
        # Pick a codeword to remove
        c_remove = random.choice(code)
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            # Removal is free
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True
        
        # Vectorized: find best replacement
        in_code = np.zeros(N, dtype=bool)
        for c in code:
            in_code[c] = True
        
        gains = cover_matrix[:, uncovered].sum(axis=1)
        gains[in_code] = 0
        
        best_c = int(np.argmax(gains))
        best_count = int(gains[best_c])
        
        if best_c != -1 and best_count == len(uncovered):
            # Single replacement works
            code.remove(c_remove)
            code.append(best_c)
            coverage[neighbors_list[c_remove]] -= 1
            coverage[neighbors_list[best_c]] += 1
            return True
        
        # Try greedy repair with multiple additions
        added = greedy_repair(set(uncovered.tolist()))
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        
        return False
>>>>>>> REPLACE

