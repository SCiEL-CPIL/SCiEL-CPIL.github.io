import time

def construct():
    start_time = time.time()
    TIMEOUT = 55.0

    def greedy_increasing_golomb():
        # a_1 = 1
        gaps = [1]
        marks = [0, 1]
        diffs = {1}
        last_gap = 1
        
        while time.time() - start_time < TIMEOUT:
            found = False
            # Next gap must be > last_gap and <= 256
            for d in range(last_gap + 1, 257):
                new_mark = marks[-1] + d
                valid = True
                for m in marks:
                    diff = new_mark - m
                    if diff in diffs:
                        valid = False
                        break
                
                if valid:
                    gaps.append(d)
                    marks.append(new_mark)
                    for m in marks[:-1]:
                        diffs.add(new_mark - m)
                    last_gap = d
                    found = True
                    break
            
            if not found:
                break
        
        return gaps

    seq = greedy_increasing_golomb()
    
    # If the greedy fails immediately or produces short sequence, 
    # we might need to try starting with a different a_1?
    # But a_1=1 is the best for length.
    # What if we skip a_1=1? No, 1 is always best.
    
    # What if we get stuck?
    # We can try to backtrack?
    # Given the tight constraints, backtracking might be expensive.
    # But the search space is small (gaps are increasing).
    # We can implement a simple DFS with pruning if greedy fails?
    # Or just return what we have.
    
    # Let's add a fallback: if len(seq) is small, try a different heuristic?
    # Actually, the greedy is likely optimal for this structure.
    
    if not seq:
        return [1]
        
    return seq
