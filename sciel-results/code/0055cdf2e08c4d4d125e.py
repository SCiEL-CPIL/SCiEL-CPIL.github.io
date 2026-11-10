    def local_search_optimized(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        
        # Initialize block counts
        block_count = [0] * npts
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                mask = pair_conflict[S_list[i]][S_list[j]]
                # Iterate set bits
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    block_count[idx] += 1
                    m ^= lsb
        
        def is_maximal_from_counts():
            for k in range(npts):
                if k not in S and block_count[k] == 0:
                    return False
            return True
        
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            for remove_p in S_list:
                if len(S) - 1 >= best_size:
                    continue
                
                # Temporarily remove remove_p
                S.remove(remove_p)
                
                # Update block counts
                # For each a in S, remove conflicts from pair_conflict[remove_p][a]
                for a in S:
                    mask = pair_conflict[remove_p][a]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        block_count[idx] -= 1
                        m ^= lsb
                
                # Check maximality
                if is_maximal_from_counts():
                    best_S = list(S)
                    best_size = len(S)
                    improved = True
                    # Keep S as is (remove_p is gone)
                    # Update best_size
                    break # Restart loop with new S
                else:
                    # Restore remove_p
                    S.add(remove_p)
                    for a in S:
                        if a != remove_p:
                            mask = pair_conflict[remove_p][a]
                            m = mask
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                block_count[idx] += 1
                                m ^= lsb
            
            if not improved:
                break
        
        return best_S
