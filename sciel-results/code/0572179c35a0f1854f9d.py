    def build_row_greedy(used, max_val):
        row = [0]
        new_diffs = set()
        for _ in range(k):
            found = False
            valid_xs = []
            # Collect valid xs first to allow random choice
            # This is slower but allows randomization
            for x in range(row[-1] + 1, max_val + 1):
                candidate_diffs = {x - a for a in row}
                if not (candidate_diffs & used) and not (candidate_diffs & new_diffs):
                    valid_xs.append((x, candidate_diffs))
            
            if not valid_xs:
                return None
            
            if random.random() < 0.8:
                # Pick smallest
                x, candidate_diffs = valid_xs[0]
            else:
                # Pick random
                x, candidate_diffs = random.choice(valid_xs)
            
            row.append(x)
            new_diffs.update(candidate_diffs)
            found = True
        
        return row, new_diffs
