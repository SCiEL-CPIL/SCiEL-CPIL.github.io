    # Helper: Greedy construction of a single row given forbidden diffs
    def greedy_row(forbidden_diffs, k, randomize=False, bias=0.1):
        row = [0]
        current_row_diffs = set()
        for _ in range(k):
            last = row[-1]
            valid_xs = []
            x = last + 1
            # Search for valid x
            # Limit search to avoid infinite loops in pathological cases, though rare for small k
            while x < 300: 
                new_diffs = {x - a for a in row}
                # Check against forbidden (other rows) AND current row diffs
                if not (new_diffs & forbidden_diffs) and not (new_diffs & current_row_diffs):
                    valid_xs.append(x)
                    if len(valid_xs) >= 10:
                        break
                x += 1
            
            if not valid_xs:
                return None
            
            if randomize:
                # Pick the smallest with high probability, otherwise random from small set
                if random.random() > bias:
                    chosen_x = valid_xs[0]
                else:
                    chosen_x = random.choice(valid_xs)
            else:
                chosen_x = valid_xs[0]
            
            row.append(chosen_x)
            # Update current_row_diffs with differences involving the new element
            current_row_diffs |= {chosen_x - a for a in row[:-1]}
        return row
