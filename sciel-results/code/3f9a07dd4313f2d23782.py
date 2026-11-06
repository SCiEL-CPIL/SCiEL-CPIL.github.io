def _local_search_first_improvement(B, deadline):
    best_B = B
    best_score = _score_bitset(B)
    
    while time.time() < deadline:
        improved = False
        bits = _bits_to_set(best_B)
        random.shuffle(bits)
        
        for x in bits:
            mask_x = ~(1 << x)
            B_without_x = best_B & mask_x
            # Precompute B_without_x to avoid re-masking
            for y in range(1, 513):
                if (best_B >> y) & 1:
                    continue
                new_B = B_without_x | (1 << y)
                new_score = _score_bitset(new_B)
                if new_score > best_score:
                    best_score = new_score
                    best_B = new_B
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return best_B, best_score
