import time
import random

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_scope = float('inf')
    best_rows = None
    
    # Bitset for used differences. 
    # Max scope estimate: 200 is safe for 4x8. 
    # We can dynamically resize or just use a large int. Python ints are arbitrary precision.
    
    def is_valid_candidate(row, x, used):
        # Check if differences x - a for a in row are all unused
        for a in row:
            d = x - a
            if (used >> d) & 1:
                return False
        return True

    def get_new_diffs(row, x):
        return [x - a for a in row]

    def run_greedy(randomize=False, random_prob=0.1):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # Find the smallest valid x
                x = last + 1
                # We might want to look a bit ahead or randomize
                # To allow randomization, we find the first few valid candidates
                candidates = []
                # Search limit: we don't want to search too far if we are randomizing
                # But for the base greedy, we find the first one.
                
                if not randomize:
                    while True:
                        if is_valid_candidate(row, x, used):
                            break
                        x += 1
                    row.append(x)
                    for a in row[:-1]: # exclude x itself? No, diffs are x-a where a in row BEFORE adding x
                         # Wait, is_valid_candidate checks against row (which doesn't have x yet)
                         # So we update used with diffs of x against all a in row
                         pass 
                    # Update used
                    for a in row[:-1]: # The row currently has x appended, so row[:-1] are the previous elements
                         d = x - a
                         used |= (1 << d)
                    # Wait, the loop above appended x to row. 
                    # Let's restructure to be clearer.
                else:
                    # Find up to 3 valid candidates to choose from
                    found = 0
                    x = last + 1
                    while found < 3:
                        if is_valid_candidate(row, x, used):
                            candidates.append(x)
                            found += 1
                        x += 1
                        if x > 200: # Safety break, shouldn't happen
                            break
                    
                    if not candidates:
                        # Fallback to smallest valid (should not happen if problem is solvable)
                        x = last + 1
                        while not is_valid_candidate(row, x, used):
                            x += 1
                        candidates = [x]
                    
                    if random.random() < random_prob and len(candidates) > 1:
                        # Pick a random candidate from the first few
                        x = random.choice(candidates)
                    else:
                        # Pick the smallest
                        x = candidates[0]
                    
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return scope, rows

    # Run base greedy first
    scope, rows = run_greedy(randomize=False)
    if scope < best_scope:
        best_scope = scope
        best_rows = rows

    # Run randomized restarts
    # Estimate time per run. The inner loop is fast.
    # Let's run as many as possible before deadline.
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Increase randomization probability slightly as we go? Or keep it constant.
        # Let's try a mix of strategies.
        # Strategy 1: Low randomization (exploit)
        # Strategy 2: High randomization (explore)
        if iteration % 5 == 0:
             p = 0.3
        else:
             p = 0.1
        
        scope, rows = run_greedy(randomize=True, random_prob=p)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            # If we found the theoretical lower bound (112), we can stop?
            # But 112 is just a bound, might not be achievable.
            # Best known is 134.
            if best_scope <= 134:
                # We beat the known best or matched it. 
                # Can we do better? Maybe. But let's keep searching a bit.
                pass

    return best_rows
