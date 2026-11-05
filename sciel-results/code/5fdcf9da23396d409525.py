        # Perturbation
        if not code: break
        idx = random.randint(0, len(code) - 1)
        c_to_remove = code[idx]
        newly_uncovered = []
        for w in ball_indices[c_to_remove]:
            cover_count[w] -= 1
            if cover_count[w] == 0:
                newly_uncovered.append(w)
        code.pop(idx)
        
        # Repair
        # We need to cover words in newly_uncovered
        # Use a set for fast lookup? Or just a list and filter.
        uncovered_set = set(newly_uncovered)
        
        while uncovered_set:
            best_c = -1
            best_score = -1
            for c in range(total_words):
                # Quick check: does c cover any word in uncovered_set?
                # We can precompute a reverse map: word -> list of codewords covering it.
                # But simpler: just iterate.
                # To speed up, we can skip if c is in code? 
                # If c is in code, its words are covered (count >= 1 from itself).
                # So if w is in uncovered_set (count 0), c cannot be in code.
                score = 0
                for w in ball_indices[c]:
                    if w in uncovered_set:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_c = c
            if best_c == -1 or best_score == 0:
                break
            code.append(best_c)
            for w in ball_indices[best_c]:
                cover_count[w] += 1
                if cover_count[w] == 1: # Was 0, now 1
                    uncovered_set.discard(w)
