import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Run multiple trials to escape local minima or find better random tie-breaks
    for trial in range(200):
        S = set()
        sums = set()
        
        while True:
            # Identify U: elements that can be added
            U = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Quick check: 2x
                if 2 * x in sums:
                    continue
                # Check x + a for a in S
                conflict = False
                for a in S:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            
            if not U:
                break
            
            # Precompute differences for O(1) conflict checks
            diffs = set()
            if S:
                S_list = list(S)
                for a in S_list:
                    for b in S_list:
                        diffs.add(b - a)
            
            # Find y in U that blocks the most others in U
            best_y = None
            max_blocked = -1
            candidates = []
            
            for y in U:
                blocked_count = 0
                for x in U:
                    if x == y:
                        continue
                    # Check conditions for x being blocked by y
                    # 1. 2x = y + a  => 2x - y in S
                    # 2. x + a = 2y  => 2y - x in S
                    # 3. x + a = y + b => x - y in diffs
                    # 4. x + y in sums
                    
                    if (2 * x - y) in S:
                        blocked_count += 1
                        continue
                    if (2 * y - x) in S:
                        blocked_count += 1
                        continue
                    if (x - y) in diffs:
                        blocked_count += 1
                        continue
                    if (x + y) in sums:
                        blocked_count += 1
                        continue
                
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_y = y
                    candidates = [y]
                elif blocked_count == max_blocked:
                    candidates.append(y)
            
            # Pick one
            # Random selection from candidates to allow exploration
            chosen_y = random.choice(candidates)
            
            # Update
            S.add(chosen_y)
            sums.add(2 * chosen_y)
            for a in S:
                if a != chosen_y:
                    sums.add(chosen_y + a)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
