    def construct():
        # ... setup ...
        best_points = []
        best_size = 0
        
        while time.time() < deadline:
            # Shuffle primes
            primes_copy = PRIMES.copy()
            random.shuffle(primes_copy)
            
            active = np.ones(RANGE, dtype=bool)
            current_size = RANGE
            
            for i, p in enumerate(primes_copy):
                if p > current_size:
                    break
                
                residues = x_indices[active] % p
                if len(residues) == 0: break
                counts = np.bincount(residues, minlength=p)
                min_count = np.min(counts)
                
                if min_count == 0:
                    continue
                
                candidate_classes = np.where(counts == min_count)[0]
                
                if len(candidate_classes) == 1:
                    r = candidate_classes[0]
                else:
                    # Lookahead
                    best_r = None
                    best_score = float('inf')
                    
                    # Get next prime
                    if i + 1 < len(primes_copy):
                        next_p = primes_copy[i+1]
                        if next_p <= current_size: # Only look ahead if next prime is active
                            for r in candidate_classes:
                                # Simulate removal of r
                                # temp_active = active & (x_indices % p != r)
                                # Instead of full mask, just compute counts for next_p
                                # We need active elements that are NOT in class r mod p
                                # active_mask_for_next = active & (x_indices % p != r)
                                # residues_next = x_indices[active_mask_for_next] % next_p
                                # counts_next = np.bincount(...)
                                # score = min(counts_next)
                                
                                # Optimization: 
                                # active elements are x where active[x]=1
                                # We want to exclude x where x%p == r
                                # We can get indices of active elements: idx = np.where(active)[0]
                                # But doing this every time is slow.
                                
                                # Let's stick to simple random if lookahead is too complex/slow.
                                pass
                    
                    # Fallback to random if lookahead not implemented or too slow
                    r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
                
                active = active & (x_indices % p != r)
                current_size -= min_count # Wait, this is wrong if we don't remove exactly min_count?
                # Yes, we remove exactly the elements in class r that are active.
                # The count of such elements is counts[r], which is min_count.
                
            points = np.where(active)[0].tolist()
            if len(points) > best_size:
                best_size = len(points)
                best_points = points
                
        return best_points
    