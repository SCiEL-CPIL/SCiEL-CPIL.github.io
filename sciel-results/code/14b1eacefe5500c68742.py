<<<<<<< SEARCH
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code
=======
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code
>>>>>>> REPLACE

