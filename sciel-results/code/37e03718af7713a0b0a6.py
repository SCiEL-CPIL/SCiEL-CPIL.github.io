<<<<<<< SEARCH
    # 3. Fast greedy using numpy vectorized gain
    def greedy_cover():
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            # Compute gain for all words at once using matrix multiply trick
            # gain[i] = number of uncovered points in neighbors[i]
            # We can use: for each i, sum of uncovered[neighbors[i]]
            # Efficient approach: for each uncovered point p, add 1 to all its neighbors' gains
            gains = np.zeros(N, dtype=np.int32)
            # For each uncovered point, increment gains of all words that cover it
            # This is the transpose operation
            uncovered_indices = np.where(uncovered)[0]
            for p in uncovered_indices:
                # Find all words that cover p: words i where p is in neighbors[i]
                # Equivalent: words i such that dist(words[i], words[p]) <= radius
                # We precompute reverse neighbors: rev[p] = list of i where p in neighbors[i]
                gains[rev_neighbors[p]] += 1
            
            # Pick the word with max gain that's not already in code
            best_idx = -1
            max_gain = 0
            for i in range(N):
                if i not in code_set and gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
            
            if max_gain == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        return code
=======
    # 3. Fast greedy using numpy vectorized gain
    def greedy_cover():
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            # Compute gain for all words at once using matrix multiply trick
            # gain[i] = number of uncovered points in neighbors[i]
            gains = np.zeros(N, dtype=np.int32)
            # For each uncovered point, increment gains of all words that cover it
            # This is the transpose operation
            uncovered_indices = np.where(uncovered)[0]
            for p in uncovered_indices:
                gains[rev_neighbors[p]] += 1
            
            # Pick the word with max gain that's not already in code
            gains[code_set] = -1
            best_idx = int(np.argmax(gains))
            max_gain = int(gains[best_idx])
            
            if max_gain <= 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        return code
>>>>>>> REPLACE

