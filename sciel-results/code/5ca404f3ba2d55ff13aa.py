<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    divisors = _get_divisors()
    n_divs = len(divisors)
    
    def best_residue_for_mod(m, uncovered_arr):
        """Find the residue r that covers the most uncovered residues for modulus m."""
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = int(np.sum(uncovered_arr[r::m]))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        return best_r, best_count
    
    def greedy_construct(div_order):
        """Greedy construction with a given divisor ordering."""
        uncovered_arr = np.ones(BASE, dtype=np.uint8)
        selected_classes = []
        used_moduli = set()
        
        for m in div_order:
            if time.time() > deadline:
                break
            if m in used_moduli:
                continue
            
            r, gain = best_residue_for_mod(m, uncovered_arr)
            if gain <= 0:
                continue
            
            used_moduli.add(m)
            selected_classes.append([m, r])
            uncovered_arr[r::m] = 0
        
        current_score = int(np.sum(uncovered_arr))
        return selected_classes, uncovered_arr, current_score
    
    def local_search(classes, uncovered_arr, current_score, max_iters=50):
        """Efficient local search: try changing each class's residue."""
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                
                m, r_old = classes[i]
                
                # Compute uncovered without class i
                unc_without = uncovered_arr.copy()
                unc_without[r_old::m] = 1  # restore what class i was covering
                
                # Find best residue for m given unc_without
                best_r = r_old
                best_gain = -1
                for r in range(m):
                    gain = int(np.sum(unc_without[r::m]))
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_r != r_old:
                    # Apply the change
                    classes[i] = [m, best_r]
                    uncovered_arr = unc_without.copy()
                    uncovered_arr[best_r::m] = 0
                    new_score = int(np.sum(uncovered_arr))
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        classes[i] = [m, r_old]
                        uncovered_arr[r_old::m] = 0
                        # uncovered_arr is now correct again
        
        return classes, uncovered_arr, current_score
    
    def add_unused_moduli(classes, uncovered_arr, current_score, divisors_sorted):
        """Try adding unused moduli that improve the score."""
        used_moduli = set(m for m, r in classes)
        improved = True
        
        while improved:
            improved = False
            for m in divisors_sorted:
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                if m in used_moduli:
                    continue
                
                best_r, best_gain = best_residue_for_mod(m, uncovered_arr)
                if best_gain > 0:
                    new_unc = uncovered_arr.copy()
                    new_unc[best_r::m] = 0
                    new_score = int(np.sum(new_unc))
                    if new_score < current_score:
                        used_moduli.add(m)
                        classes.append([m, best_r])
                        uncovered_arr = new_unc
                        current_score = new_score
                        improved = True
        
        return classes, uncovered_arr, current_score
    
    def swap_local_search(classes, uncovered_arr, current_score, max_iters=20):
        """Try swapping residues between pairs of classes."""
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                for j in range(i + 1, len(classes)):
                    if time.time() > deadline:
                        return classes, uncovered_arr, current_score
                    
                    mi, ri = classes[i]
                    mj, rj = classes[j]
                    
                    # Compute uncovered without classes i and j
                    unc_ij = uncovered_arr.copy()
                    unc_ij[ri::mi] = 1
                    unc_ij[rj::mj] = 1
                    
                    # Try all residue pairs
                    best_ri = ri
                    best_rj = rj
                    best_score = current_score
                    
                    # Only try a subset for efficiency
                    ri_candidates = [ri]
                    for r in range(min(mi, 50)):
                        if r != ri:
                            ri_candidates.append(r)
                    rj_candidates = [rj]
                    for r in range(min(mj, 50)):
                        if r != rj:
                            rj_candidates.append(r)
                    
                    for new_ri in ri_candidates:
                        for new_rj in rj_candidates:
                            if new_ri == ri and new_rj == rj:
                                continue
                            
                            test_unc = unc_ij.copy()
                            test_unc[new_ri::mi] = 0
                            test_unc[new_rj::mj] = 0
                            test_score = int(np.sum(test_unc))
                            
                            if test_score < best_score:
                                best_score = test_score
                                best_ri = new_ri
                                best_rj = new_rj
                    
                    if best_ri != ri or best_rj != rj:
                        classes[i] = [mi, best_ri]
                        classes[j] = [mj, best_rj]
                        uncovered_arr = unc_ij.copy()
                        uncovered_arr[best_ri::mi] = 0
                        uncovered_arr[best_rj::mj] = 0
                        current_score = best_score
                        improved = True
        
        return classes, uncovered_arr, current_score
    
    # Try multiple greedy orderings
    divisors_sorted = sorted(divisors)
    divisors_desc = sorted(divisors, reverse=True)
    
    # Ordering 1: ascending (small moduli first)
    classes1, unc1, score1 = greedy_construct(divisors_sorted)
    classes1, unc1, score1 = local_search(classes1, unc1, score1, max_iters=30)
    classes1, unc1, score1 = add_unused_moduli(classes1, unc1, score1, divisors_sorted)
    classes1, unc1, score1 = local_search(classes1, unc1, score1, max_iters=30)
    
    # Ordering 2: descending (large moduli first)
    if time.time() < deadline:
        classes2, unc2, score2 = greedy_construct(divisors_desc)
        classes2, unc2, score2 = local_search(classes2, unc2, score2, max_iters=30)
        classes2, unc2, score2 = add_unused_moduli(classes2, unc2, score2, divisors_sorted)
        classes2, unc2, score2 = local_search(classes2, unc2, score2, max_iters=30)
    else:
        classes2, unc2, score2 = classes1, unc1, score1
    
    # Pick the best
    if score2 < score1:
        best_classes, best_unc, best_score = classes2, unc2, score2
    else:
        best_classes, best_unc, best_score = classes1, unc1, score1
    
    # Final polish: swap-based local search
    if time.time() < deadline:
        best_classes, best_unc, best_score = swap_local_search(
            best_classes, best_unc, best_score, max_iters=15
        )
    
    # One more round of standard local search
    if time.time() < deadline:
        best_classes, best_unc, best_score = local_search(
            best_classes, best_unc, best_score, max_iters=20
        )
    
    # Final add pass
    if time.time() < deadline:
        best_classes, best_unc, best_score = add_unused_moduli(
            best_classes, best_unc, best_score, divisors_sorted
        )
    
    return [[m, r] for m, r in best_classes]
=======
def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit
    
    divisors = _get_divisors()
    masks = _get_masks()
    n_divs = len(divisors)
    
    def best_residue_for_mod(m, uncovered_arr):
        """Find the residue r that covers the most uncovered residues for modulus m."""
        best_r = 0
        best_count = -1
        m_masks = masks[m]
        for r in range(m):
            cnt = int(np.sum(uncovered_arr & m_masks[r]))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        return best_r, best_count
    
    def greedy_construct(div_order):
        """Greedy construction with a given divisor ordering."""
        uncovered_arr = np.ones(BASE, dtype=np.uint8)
        selected_classes = []
        used_moduli = set()
        
        for m in div_order:
            if time.time() > deadline:
                break
            if m in used_moduli:
                continue
            
            r, gain = best_residue_for_mod(m, uncovered_arr)
            if gain <= 0:
                continue
            
            used_moduli.add(m)
            selected_classes.append([m, r])
            uncovered_arr &= ~masks[m][r]
        
        current_score = int(np.sum(uncovered_arr))
        return selected_classes, uncovered_arr, current_score
    
    def local_search(classes, uncovered_arr, current_score, max_iters=50):
        """Efficient local search: try changing each class's residue."""
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                
                m, r_old = classes[i]
                
                # Compute uncovered without class i
                unc_without = uncovered_arr | masks[m][r_old]
                
                # Find best residue for m given unc_without
                best_r = r_old
                best_gain = -1
                m_masks = masks[m]
                for r in range(m):
                    gain = int(np.sum(unc_without & m_masks[r]))
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_r != r_old:
                    # Apply the change
                    classes[i] = [m, best_r]
                    uncovered_arr = unc_without & ~masks[m][best_r]
                    new_score = int(np.sum(uncovered_arr))
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        classes[i] = [m, r_old]
                        uncovered_arr = unc_without & ~masks[m][r_old]
        
        return classes, uncovered_arr, current_score
    
    def add_unused_moduli(classes, uncovered_arr, current_score, divisors_sorted):
        """Try adding unused moduli that improve the score."""
        used_moduli = set(m for m, r in classes)
        improved = True
        
        while improved:
            improved = False
            for m in divisors_sorted:
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                if m in used_moduli:
                    continue
                
                best_r, best_gain = best_residue_for_mod(m, uncovered_arr)
                if best_gain > 0:
                    new_unc = uncovered_arr & ~masks[m][best_r]
                    new_score = int(np.sum(new_unc))
                    if new_score < current_score:
                        used_moduli.add(m)
                        classes.append([m, best_r])
                        uncovered_arr = new_unc
                        current_score = new_score
                        improved = True
        
        return classes, uncovered_arr, current_score
    
    def swap_local_search(classes, uncovered_arr, current_score, max_iters=20):
        """Try swapping residues between pairs of classes."""
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                for j in range(i + 1, len(classes)):
                    if time.time() > deadline:
                        return classes, uncovered_arr, current_score
                    
                    mi, ri = classes[i]
                    mj, rj = classes[j]
                    
                    # Compute uncovered without classes i and j
                    unc_ij = uncovered_arr | masks[mi][ri] | masks[mj][rj]
                    
                    # Try all residue pairs
                    best_ri = ri
                    best_rj = rj
                    best_score = current_score
                    
                    # Try all residues for both (moduli are small enough)
                    for new_ri in range(mi):
                        for new_rj in range(mj):
                            if new_ri == ri and new_rj == rj:
                                continue
                            
                            test_unc = unc_ij & ~masks[mi][new_ri] & ~masks[mj][new_rj]
                            test_score = int(np.sum(test_unc))
                            
                            if test_score < best_score:
                                best_score = test_score
                                best_ri = new_ri
                                best_rj = new_rj
                    
                    if best_ri != ri or best_rj != rj:
                        classes[i] = [mi, best_ri]
                        classes[j] = [mj, best_rj]
                        uncovered_arr = unc_ij & ~masks[mi][best_ri] & ~masks[mj][best_rj]
                        current_score = best_score
                        improved = True
        
        return classes, uncovered_arr, current_score
    
    # Try multiple greedy orderings
    divisors_sorted = sorted(divisors)
    divisors_desc = sorted(divisors, reverse=True)
    
    # Ordering 1: ascending (small moduli first)
    classes1, unc1, score1 = greedy_construct(divisors_sorted)
    classes1, unc1, score1 = local_search(classes1, unc1, score1, max_iters=30)
    classes1, unc1, score1 = add_unused_moduli(classes1, unc1, score1, divisors_sorted)
    classes1, unc1, score1 = local_search(classes1, unc1, score1, max_iters=30)
    
    # Ordering 2: descending (large moduli first)
    if time.time() < deadline:
        classes2, unc2, score2 = greedy_construct(divisors_desc)
        classes2, unc2, score2 = local_search(classes2, unc2, score2, max_iters=30)
        classes2, unc2, score2 = add_unused_moduli(classes2, unc2, score2, divisors_sorted)
        classes2, unc2, score2 = local_search(classes2, unc2, score2, max_iters=30)
    else:
        classes2, unc2, score2 = classes1, unc1, score1
    
    # Pick the best
    if score2 < score1:
        best_classes, best_unc, best_score = classes2, unc2, score2
    else:
        best_classes, best_unc, best_score = classes1, unc1, score1
    
    # Final polish: swap-based local search
    if time.time() < deadline:
        best_classes, best_unc, best_score = swap_local_search(
            best_classes, best_unc, best_score, max_iters=15
        )
    
    # One more round of standard local search
    if time.time() < deadline:
        best_classes, best_unc, best_score = local_search(
            best_classes, best_unc, best_score, max_iters=20
        )
    
    # Final add pass
    if time.time() < deadline:
        best_classes, best_unc, best_score = add_unused_moduli(
            best_classes, best_unc, best_score, divisors_sorted
        )
    
    return [[m, r] for m, r in best_classes]
>>>>>>> REPLACE
