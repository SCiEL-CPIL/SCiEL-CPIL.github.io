import time
import random
import math

def construct(n_rows, k):
    # n_rows = 4, k = 7 (implied by "8 increasing integers")
    # The prompt says "4 rows of 8 increasing integers", so k=7 in the formula a_0..a_k
    # But the interface is construct(n_rows, k). 
    # If the evaluator passes k=7, we generate k+1=8 elements.
    # If the evaluator passes k=8, we generate 9? 
    # "Choose 4 rows of 8 increasing integers" -> 8 elements.
    # The seed code: while len(row) < k + 1.
    # So if the task is "8 integers", k must be 7.
    # I will assume the evaluator passes the correct k for the task.
    
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    # Initialize best solution
    # We can start with a greedy solution or random
    best_solution = None
    best_score = float('inf')
    
    # Helper to check validity and get score
    def get_score_and_diffs(rows):
        used = set()
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return None, None # Invalid
                    used.add(d)
            max_val = max(max_val, r[-1])
        return max_val, used

    # Initial greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    init_rows = greedy_init()
    score, _ = get_score_and_diffs(init_rows)
    if score is not None:
        best_solution = [list(r) for r in init_rows]
        best_score = score

    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.999
    current_rows = [list(r) for r in best_solution]
    current_score = best_score
    
    # Precompute differences for current rows for faster updates?
    # Since k is small, recomputing is fine, but maintaining a set is better.
    
    # We need to be careful with the "distinct differences" constraint.
    # Changing one element affects k differences in that row.
    
    while time.time() < deadline:
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k) # 1 to k (0 is fixed to 0)
        
        old_val = current_rows[r_idx][p_idx]
        
        # Determine range for new value
        lower = current_rows[r_idx][p_idx - 1] + 1
        # If p_idx < k, upper is current_rows[r_idx][p_idx + 1] - 1
        # If p_idx == k, upper can be large, but we want to minimize max, so maybe limit to current max + some noise
        if p_idx < k:
            upper = current_rows[r_idx][p_idx + 1] - 1
        else:
            # Allow expansion but biased towards current max
            upper = current_score + random.randint(0, 10)
            
        if lower > upper:
            continue
            
        # Proposal: pick a new value
        # Bias towards values that keep the row compact
        new_val = random.randint(lower, upper)
        if new_val == old_val:
            continue
            
        # Calculate new score and validity
        # To speed up, we can check validity locally
        # But full check is O(n_rows * k^2) which is small (4 * 28 * 2 ~ 224 ops)
        # So full check is fine.
        
        # Save old value
        current_rows[r_idx][p_idx] = new_val
        
        new_score, _ = get_score_and_diffs(current_rows)
        
        if new_score is not None:
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_solution = [list(r) for r in current_rows]
            else:
                # Accept with probability exp(-(delta)/T)
                delta = new_score - current_score
                if delta == 0:
                    # Accept equal score to explore
                    if random.random() < 0.5:
                        pass # keep new
                    else:
                        current_rows[r_idx][p_idx] = old_val
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        pass # keep new
                    else:
                        current_rows[r_idx][p_idx] = old_val
        else:
            # Invalid, revert
            current_rows[r_idx][p_idx] = old_val
            
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            # Restart with higher T if stuck? Or just continue
            # Let's do a soft restart: randomize a few elements
            if random.random() < 0.01:
                # Perturb
                for _ in range(3):
                    ri = random.randint(0, n_rows - 1)
                    pi = random.randint(1, k)
                    lo = current_rows[ri][pi-1] + 1
                    hi = current_rows[ri][pi+1] - 1 if pi < k else current_score + 5
                    if lo <= hi:
                        current_rows[ri][pi] = random.randint(lo, hi)
                # Check validity after perturbation
                s, _ = get_score_and_diffs(current_rows)
                if s is None:
                    # Revert to best
                    current_rows = [list(r) for r in best_solution]
                    current_score = best_score
                else:
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_solution = [list(r) for r in current_rows]

    return best_solution
