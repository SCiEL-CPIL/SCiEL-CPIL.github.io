import random
import math
import time

def construct(n_rows, k):
    # Target: n_rows rows, each of length k+1
    # Approach: Simulated Annealing
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            if any(r[i] >= r[i+1] for i in range(len(r)-1)):
                return False
            r_diffs = get_row_diffs(r)
            if len(r_diffs) != len(r)*(len(r)-1)//2:
                return False
            if all_diffs & r_diffs:
                return False
            all_diffs |= r_diffs
        return True

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initial solution
    rows = greedy_construct()
    current_score = max(r[-1] for r in rows)
    best_rows = [r[:] for r in rows]
    best_score = current_score
    
    start_time = time.time()
    timeout = 110.0
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 1.0
    alpha = 0.995
    max_iter = 100000
    
    # Precompute diffs for efficiency
    current_diffs = [get_row_diffs(r) for r in rows]
    current_all_diffs = set()
    for d in current_diffs:
        current_all_diffs |= d
        
    # Helper to check if a new row is valid against others
    # row_diffs: set of diffs for the candidate row
    # other_diffs: set of diffs for all other rows
    # We need to ensure row_diffs is disjoint from other_diffs
    # And row_diffs has correct size (implied by construction if we check strictly increasing)
    
    def try_move(row_idx, pos, new_val):
        # Returns (new_row, new_diffs) if valid, else None
        # We modify row_idx at position pos to new_val
        # Must ensure strict increase
        old_row = rows[row_idx]
        if pos > 0 and new_val <= old_row[pos-1]:
            return None
        if pos < k and new_val >= old_row[pos+1]:
            return None
        
        new_row = old_row[:]
        new_row[pos] = new_val
        
        # Check internal validity (strictly increasing is handled above, but let's be safe)
        # Actually, if we just change one value, we just need to check diffs involving that value
        # But it's safer to recompute diffs for the row if it's small (k=5, len=6)
        new_diffs = get_row_diffs(new_row)
        
        # Check disjointness with other rows
        # other_diffs = current_all_diffs - current_diffs[row_idx]
        # We can precompute or just check against the union minus current row's diffs
        # For simplicity and speed in Python, let's just check intersection with a precomputed "others" set?
        # No, "others" changes if we update.
        # Let's just check: new_diffs & (current_all_diffs - current_diffs[row_idx]) == empty
        others = current_all_diffs - current_diffs[row_idx]
        if new_diffs & others:
            return None
        
        # Also check internal distinctness (should be fine if strictly increasing integers? No, 0,1,3 -> diffs 1,2,3. 0,1,2 -> 1,1,2. Invalid.)
        # get_row_diffs handles the set size check implicitly if we assume distinctness?
        # The task requires "all differences ... are distinct over all rows".
        # This implies within a row they must be distinct too?
        # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
        # Yes, global distinctness.
        # If new_diffs has size < k*(k+1)/2, it has duplicates.
        if len(new_diffs) != k*(k+1)//2:
            return None
            
        return new_row, new_diffs

    for _ in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        if T < T_min:
            break
            
        # Select a random row and a random position to perturb
        row_idx = random.randint(0, n_rows - 1)
        # We can perturb any position 1..k (index 0 is fixed at 0)
        pos = random.randint(1, k)
        
        old_row = rows[row_idx]
        old_val = old_row[pos]
        
        # Propose a new value
        # Random walk around current value
        # Step size can be adaptive or fixed
        step = random.randint(-5, 5)
        if step == 0:
            step = 1
        new_val = old_val + step
        
        # Ensure bounds
        min_allowed = old_row[pos-1] + 1 if pos > 0 else 1
        max_allowed = old_row[pos+1] - 1 if pos < k else current_score + 10 # Allow growing if stuck?
        
        # Actually, to minimize max, we should try to shrink.
        # But SA allows growth.
        # Let's constrain new_val to be reasonable.
        if new_val < min_allowed:
            new_val = min_allowed
        if new_val > max_allowed:
            # If we hit the upper bound, maybe try to expand the row?
            # For now, just clamp or skip
            if pos == k:
                 # Last element, can grow
                 new_val = max_allowed
            else:
                 # Middle element, constrained by next element
                 new_val = max_allowed
                 
        if new_val == old_val:
            continue
            
        result = try_move(row_idx, pos, new_val)
        if not result:
            T *= alpha
            continue
            
        new_row, new_diffs = result
        new_max = new_row[-1]
        
        # Calculate score change
        # Score is max(max(row) for row in rows)
        # If we change row_idx, the new score is max(old_score, new_max) IF row_idx was the max row
        # Or if new_max > old_score, new score is new_max.
        # If new_max < old_score and row_idx was the unique max row, new score is max of others.
        
        # To keep it simple, let's define score as sum of maxes? No, task says "largest entry as small as possible".
        # So score = max(r[-1] for r in rows).
        
        old_score = current_score
        # Update current state temporarily to calculate new score
        # We need to know if row_idx was the bottleneck.
        # Let's just calculate new_score by taking max of all rows with the new one.
        # This is O(n). n is small (7).
        
        # Temporarily swap
        old_diffs = current_diffs[row_idx]
        
        # Check if move improves or is accepted
        # Calculate new global max
        # We can do this efficiently:
        # If new_max <= old_score:
        #   new_score = old_score (unless row_idx was the ONLY row with max=old_score, then it might drop)
        #   Actually, if row_idx[-1] == old_score and new_max < old_score, new_score is max of other rows.
        #   If row_idx[-1] < old_score, new_score is old_score.
        # If new_max > old_score:
        #   new_score = new_max
        
        if new_max > old_score:
            new_score = new_max
        else:
            if old_row[-1] == old_score:
                # Find max of other rows
                max_other = 0
                for j in range(n_rows):
                    if j != row_idx:
                        if rows[j][-1] > max_other:
                            max_other = rows[j][-1]
                new_score = max_other
            else:
                new_score = old_score
                
        delta = new_score - old_score
        
        # Acceptance
        if delta < 0:
            # Improvement
            # Apply move
            rows[row_idx] = new_row
            current_diffs[row_idx] = new_diffs
            
            # Update current_all_diffs
            current_all_diffs -= old_diffs
            current_all_diffs |= new_diffs
            
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [r[:] for r in rows]
                
            # Lower temperature slightly faster on improvement? Or keep schedule.
            T *= alpha
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            if delta == 0:
                # Neutral, accept to explore
                pass
            else:
                # If delta is large, probability is low
                p = math.exp(-delta / T)
                if random.random() > p:
                    # Reject
                    T *= alpha
                    continue
                # If we reject, we don't apply the move
                T *= alpha
                continue
            # If we reach here, we accept the worse move (or neutral)
            # Wait, the logic above is mixed.
            # Let's restructure acceptance.
            
    # The loop above has a logic flaw in the acceptance branch. Let's rewrite the main loop cleanly.
    
    # RESTART with clean SA loop
    
    rows = greedy_construct()
    current_score = max(r[-1] for r in rows)
    best_rows = [r[:] for r in rows]
    best_score = current_score
    current_diffs = [get_row_diffs(r) for r in rows]
    current_all_diffs = set()
    for d in current_diffs:
        current_all_diffs |= d
        
    start_time = time.time()
    timeout = 110.0
    T = 50.0
    T_min = 1.0
    alpha = 0.995
    
    while T > T_min and (time.time() - start_time) < timeout:
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        old_row = rows[row_idx]
        old_val = old_row[pos]
        
        # Propose move
        # Try to move towards smaller values if possible, or random
        # Bias towards shrinking
        if random.random() < 0.7:
            step = random.randint(-3, 1)
        else:
            step = random.randint(-5, 5)
            
        new_val = old_val + step
        
        # Bounds check
        min_allowed = old_row[pos-1] + 1
        max_allowed = old_row[pos+1] - 1 if pos < k else old_val + 5 # Allow slight growth at end
        
        if new_val < min_allowed:
            new_val = min_allowed
        if new_val > max_allowed:
            new_val = max_allowed
            
        if new_val == old_val:
            continue
            
        # Validate
        new_row = old_row[:]
        new_row[pos] = new_val
        
        # Check strict increase (already handled by bounds, but double check)
        if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
            continue
            
        new_diffs = get_row_diffs(new_row)
        if len(new_diffs) != k*(k+1)//2:
            continue
            
        others = current_all_diffs - current_diffs[row_idx]
        if new_diffs & others:
            continue
            
        # Calculate Score
        new_max = new_row[-1]
        old_score = current_score
        
        if new_max > old_score:
            new_score = new_max
        else:
            if old_row[-1] == old_score:
                max_other = 0
                for j in range(n_rows):
                    if j != row_idx:
                        if rows[j][-1] > max_other:
                            max_other = rows[j][-1]
                new_score = max_other
            else:
                new_score = old_score
                
        delta = new_score - old_score
        
        # Acceptance
        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > 0:
                p = math.exp(-delta / T)
                if random.random() < p:
                    accept = True
                    
        if accept:
            # Apply
            old_diffs = current_diffs[row_idx]
            rows[row_idx] = new_row
            current_diffs[row_idx] = new_diffs
            current_all_diffs -= old_diffs
            current_all_diffs |= new_diffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [r[:] for r in rows]
                
        T *= alpha
        
    return best_rows
