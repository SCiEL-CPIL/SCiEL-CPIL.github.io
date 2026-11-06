import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    MAX_VAL = 512
    N = 32
    MAX_D = MAX_VAL - 1  # 511
    
    rng = random.Random(42)
    
    # Boolean array for fast membership
    present = np.zeros(MAX_VAL + 1, dtype=bool)
    
    # Start with a random set
    current_set = set(rng.sample(range(1, MAX_VAL + 1), N))
    for x in current_set:
        present[x] = True
    
    # count_d[d] = number of 3-APs with difference d in the set
    count_d = np.zeros(MAX_D + 1, dtype=np.int32)
    
    def compute_count_d():
        """Compute count_d from scratch."""
        count_d[:] = 0
        S = list(current_set)
        for x in S:
            # x is the start: x, x+d, x+2d
            max_d_start = (MAX_VAL - x) // 2
            for d in range(1, max_d_start + 1):
                if present[x + d] and present[x + 2 * d]:
                    count_d[d] += 1
            # x is the middle: x-d, x, x+d
            max_d_mid = min(x - 1, MAX_VAL - x)
            for d in range(1, max_d_mid + 1):
                if present[x - d] and present[x + d]:
                    count_d[d] += 1
            # x is the end: x-2d, x-d, x
            max_d_end = x // 2
            for d in range(1, max_d_end + 1):
                if present[x - 2 * d] and present[x - d]:
                    count_d[d] += 1
    
    compute_count_d()
    current_score = np.count_nonzero(count_d)
    best_score = current_score
    best_set = set(current_set)
    best_count_d = count_d.copy()
    
    # SAT-inspired local search
    # Focus on uncovered differences: prioritize moves that cover new differences
    steps_since_improvement = 0
    max_steps_no_improve = 2000
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if steps_since_improvement >= max_steps_no_improve:
            # Reset to best and try a different approach
            current_set = set(best_set)
            present[:] = False
            for x in current_set:
                present[x] = True
            count_d[:] = best_count_d
            current_score = best_score
            steps_since_improvement = 0
            # Perturb: random swap
            for _ in range(5):
                x_out = rng.choice(list(current_set))
                y_in = rng.randint(1, MAX_VAL)
                if y_in not in current_set:
                    current_set.remove(x_out)
                    present[x_out] = False
                    current_set.add(y_in)
                    present[y_in] = True
            compute_count_d()
            current_score = np.count_nonzero(count_d)
            continue
        
        # Compute delta for a swap
        x_out = rng.choice(list(current_set))
        y_in = rng.randint(1, MAX_VAL)
        if y_in in current_set:
            continue
        
        # Calculate score change
        # Remove x_out, add y_in
        # We need to compute how count_d changes
        
        # For each d, track if it was covered before and after
        # This is O(511) per swap evaluation
        
        # Quick delta computation
        old_score = current_score
        
        # Temporarily remove x_out
        present[x_out] = False
        # Count APs involving x_out that will be lost
        lost = {}
        for d in range(1, MAX_D + 1):
            c = 0
            # x_out as start
            if x_out + 2 * d <= MAX_VAL and present[x_out + d] and present[x_out + 2 * d]:
                c += 1
            # x_out as middle
            if x_out - d >= 1 and x_out + d <= MAX_VAL and present[x_out - d] and present[x_out + d]:
                c += 1
            # x_out as end
            if x_out - 2 * d >= 1 and present[x_out - 2 * d] and present[x_out - d]:
                c += 1
            if c > 0:
                lost[d] = c
        
        # Temporarily add y_in
        present[y_in] = True
        # Count APs involving y_in that will be gained
        gained = {}
        for d in range(1, MAX_D + 1):
            c = 0
            # y_in as start
            if y_in + 2 * d <= MAX_VAL and present[y_in + d] and present[y_in + 2 * d]:
                c += 1
            # y_in as middle
            if y_in - d >= 1 and y_in + d <= MAX_VAL and present[y_in - d] and present[y_in + d]:
                c += 1
            # y_in as end
            if y_in - 2 * d >= 1 and present[y_in - 2 * d] and present[y_in - d]:
                c += 1
            if c > 0:
                gained[d] = c
        
        # Restore present
        present[x_out] = True
        present[y_in] = False
        
        # Compute new score
        new_score = old_score
        # For each d, check if coverage changes
        all_d = set(lost.keys()) | set(gained.keys())
        for d in all_d:
            was_covered = count_d[d] > 0
            # After swap: count_d[d] - lost.get(d,0) + gained.get(d,0)
            new_count = count_d[d] - lost.get(d, 0) + gained.get(d, 0)
            now_covered = new_count > 0
            if was_covered and not now_covered:
                new_score -= 1
            elif not was_covered and now_covered:
                new_score += 1
        
        # Accept with Metropolis-like probability
        delta = new_score - old_score
        if delta >= 0:
            accept = True
        else:
            # Probability of accepting a worse move
            accept = rng.random() < np.exp(delta / temperature)
        
        if accept:
            # Apply swap
            current_set.remove(x_out)
            present[x_out] = False
            current_set.add(y_in)
            present[y_in] = True
            
            # Update count_d
            for d in all_d:
                count_d[d] += gained.get(d, 0) - lost.get(d, 0)
            
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                best_count_d = count_d.copy()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            temperature *= cooling
        else:
            steps_since_improvement += 1
    
    return sorted(best_set)
