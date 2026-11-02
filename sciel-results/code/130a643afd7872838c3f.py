import math
import time

def construct():
    # Search for a small set that maximizes the score
    # Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    
    def get_score(U):
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        # Compute sumset and diffset
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 1:
            return 1.0 # Lower bound
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1 + math.log(ratio) / denom

    # Initialize with a few good candidates
    candidates = [
        [0, 1, 4],
        [0, 1, 5],
        [0, 1, 6],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 2, 3, 4],
        [0, 1, 4, 6],
        [0, 1, 5, 7],
        [0, 1, 3, 6, 9]
    ]
    
    best_score = 0
    best_set = [0, 1, 4]
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Simple local search: try adding/removing/swapping elements
    # Keep it fast to fit within time limits
    current = list(best_set)
    for _ in range(100): # 100 iterations
        # Try adding a small random number
        if len(current) < 10:
            new_set = current + [max(current) + 1]
            s = get_score(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                current = list(best_set)
                continue
                
        # Try removing an element (if size > 2)
        if len(current) > 2:
            for i in range(len(current)):
                new_set = current[:i] + current[i+1:]
                if 0 in new_set:
                    s = get_score(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        current = list(best_set)
                        break
                        
        # Try swapping an element
        if len(current) > 1:
            for i in range(1, len(current)):
                for delta in [-1, 1, -2, 2]:
                    new_val = current[i] + delta
                    if new_val > 0 and new_val not in current:
                        new_set = current[:i] + [new_val] + current[i+1:]
                        if 0 in new_set:
                            s = get_score(new_set)
                            if s > best_score:
                                best_score = s
                                best_set = new_set
                                current = list(best_set)
                                break
    
    return sorted(best_set)
