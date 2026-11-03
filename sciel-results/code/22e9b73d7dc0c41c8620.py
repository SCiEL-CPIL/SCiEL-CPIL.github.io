    def find_row(used, limit, k, max_steps=10000):
        row = [0]
        steps = [0]
        
        def rec(idx):
            if steps[0] > max_steps:
                return False
            steps[0] += 1
            
            if idx == k + 1:
                return True
            
            min_next = row[-1] + 1
            max_next = limit
            
            if min_next > max_next:
                return False
                
            candidates = list(range(min_next, max_next + 1))
            random.shuffle(candidates)
            
            for v in candidates:
                new_diffs = {v - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used:
                    continue
                
                row.append(v)
                used.update(new_diffs)
                
                if rec(idx + 1):
                    return True
                    
                used.difference_update(new_diffs)
                row.pop()
                
            return False

        if rec(0):
            return row
        return None
