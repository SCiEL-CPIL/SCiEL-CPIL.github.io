<<<<<<< SEARCH
    # Try top exponent sets
    for ae, exps in scored_exps[:300]:
        if time.time() > deadline:
            break
        
        # First try sign-only optimization (fast)
        score, signs = optimize_signs(exps, max_iter=30)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, signs)
            if candidate is not None:
                best_candidate = candidate
        
        # If we have time, also try coefficient optimization
        if time.time() < deadline - 5:
            score2, coeffs = optimize_coeffs(exps, max_iter=20)
            
            if score2 < best_score:
                best_score = score2
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_candidate = candidate
=======
    # Try top exponent sets
    for ae, exps in scored_exps[:300]:
        if time.time() > deadline:
            break
        
        # First try sign-only optimization (fast)
        score, signs = optimize_signs(exps, max_iter=30)
        
        if score < best_score:
            best_score = score
            candidate = _build_coeff_list(exps, signs)
            if candidate is not None:
                best_candidate = candidate
        
        # If we have time, also try coefficient optimization
        if time.time() < deadline - 5:
            score2, coeffs = optimize_coeffs(exps, max_iter=20)
            
            if score2 < best_score:
                best_score = score2
                candidate = _build_coeff_list(exps, coeffs)
                if candidate is not None:
                    best_candidate = candidate
        
        # Try multi-flip sign optimization for better local optima escape
        if time.time() < deadline - 10 and score <= best_score + 2:
            def optimize_signs_multiflip(exps, max_iter=50):
                """Local search with multi-flip moves."""
                import random
                rng = random.Random(hash(tuple(exps)) % 10000)
                signs = [1] * 12
                current_score = _square_nonzero_count(exps, signs)
                
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    improved = False
                    # Single flip
                    for i in range(12):
                        signs[i] = -signs[i]
                        new_score = _square_nonzero_count(exps, signs)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            signs[i] = -signs[i]
                    # Double flip
                    for i in range(12):
                        for j in range(i + 1, 12):
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            new_score = _square_nonzero_count(exps, signs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                signs[i] = -signs[i]
                                signs[j] = -signs[j]
                    # Triple flip (less frequent)
                    if not improved and rng.random() < 0.3:
                        for _t in range(20):
                            i, j, k = rng.sample(range(12), 3)
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
                            new_score = _square_nonzero_count(exps, signs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                signs[i] = -signs[i]
                                signs[j] = -signs[j]
                                signs[k] = -signs[k]
                    if not improved:
                        break
                return current_score, signs
            
            score3, signs3 = optimize_signs_multiflip(exps, max_iter=30)
            if score3 < best_score:
                best_score = score3
                candidate = _build_coeff_list(exps, signs3)
                if candidate is not None:
                    best_candidate = candidate
>>>>>>> REPLACE

