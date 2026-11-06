<<<<<<< SEARCH
    def local_improve_np(classes, deadline):
=======
    def polish_residues(classes, deadline):
        """
        Improve the solution by adjusting residues of existing primes.
        Greedy: Find first gap n. Try to assign a residue to any prime p 
        such that n is covered. Accept if L increases.
        """
        nonlocal best_L, best_classes
        
        if not classes:
            return
            
        curr_classes = [list(c) for c in classes]
        
        counts = np.zeros(N + 1, dtype=np.int16)
        
        def rebuild_counts(cl):
            nonlocal counts
            counts = np.zeros(N + 1, dtype=np.int16)
            for p, r in cl:
                start_k = r if r != 0 else p
                idxs = np.arange(start_k, N + 1, p, dtype=np.int32)
                counts[idxs] += 1
            
        rebuild_counts(curr_classes)
        
        def get_L_from_counts(cnts):
            zeros = np.nonzero(cnts[1:] == 0)[0]
            if len(zeros) == 0:
                return N
            return int(zeros[0])
            
        curr_L = get_L_from_counts(counts)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if curr_L >= N:
                break
            n_gap = curr_L + 1
            
            best_new_L = curr_L
            best_change = None # (index, new_r)
            
            for i, (p, r_old) in enumerate(curr_classes):
                r_new = n_gap % p
                if r_new == r_old:
                    continue
                
                start_old = r_old if r_old != 0 else p
                idxs_old = np.arange(start_old, N + 1, p, dtype=np.int32)
                counts[idxs_old] -= 1
                
                start_new = r_new if r_new != 0 else p
                idxs_new = np.arange(start_new, N + 1, p, dtype=np.int32)
                counts[idxs_new] += 1
                
                slice_counts = counts[n_gap:]
                zeros = np.nonzero(slice_counts == 0)[0]
                if len(zeros) == 0:
                    new_L = N
                else:
                    new_L = n_gap + int(zeros[0])
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_change = (i, r_new)
                
                counts[idxs_old] += 1
                counts[idxs_new] -= 1
            
            if best_change is not None:
                i, r_new = best_change
                curr_classes[i][1] = r_new
                curr_L = best_new_L
                improved = True
                if curr_L > best_L:
                    best_L = curr_L
                    best_classes = [list(c) for c in curr_classes]

    def local_improve_np(classes, deadline):
>>>>>>> REPLACE

