import time
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for the evaluator's 60s/90s limit

    # 1. Greedy Baseline
    # This provides a good initial best_len and best_seq for pruning.
    values, sums = [], set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values[:]
    best_len = len(values)
    
    # 2. DFS to maximize length with pruning
    
    def max_additional_gaps(last_gap, remaining_space):
        """
        Calculate the maximum number of additional gaps we can fit
        given the last gap and remaining space, assuming gaps are 
        last_gap+1, last_gap+2, ...
        """
        if remaining_space <= 0:
            return 0
        
        # Solve m(m + 2g + 1) <= 2 * remaining_space
        # m^2 + m(2g+1) - 2R <= 0
        # m = (- (2g+1) + sqrt((2g+1)^2 + 8R)) / 2
        
        b = 2 * last_gap + 1
        c = -2 * remaining_space
        disc = b*b - 4*c
        if disc < 0:
            return 0
        m_max = int((-b + math.sqrt(disc)) / 2)
        
        # Verify and adjust
        while m_max > 0:
            s = m_max * (last_gap + 1) + m_max * (m_max - 1) // 2
            if s <= remaining_space:
                return m_max
            m_max -= 1
        return 0

    def dfs(current_seq, last_gap, current_sum, used_sums):
        nonlocal best_seq, best_len, deadline

        if time.time() > deadline:
            return

        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]

        remaining_space = 256 - current_sum
        if remaining_space <= 0:
            return

        max_add = max_additional_gaps(last_gap, remaining_space)
        if len(current_seq) + max_add <= best_len:
            return

        start_x = last_gap + 1
        end_x = min(256 - current_sum, 256)
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                return

            valid = True
            new_sums = []
            t = x
            for a in reversed(current_seq):
                new_sums.append(t)
                t += a
            new_sums.append(t)
            
            for s in new_sums:
                if s in used_sums:
                    valid = False
                    break
            
            if not valid:
                continue

            used_sums.update(new_sums)
            current_seq.append(x)
            
            dfs(current_seq, x, current_sum + x, used_sums)
            
            current_seq.pop()
            for s in new_sums:
                used_sums.remove(s)

    # Start DFS from empty to potentially find a better structure than greedy
    dfs([], 0, 0, set())
    
    return best_seq

construct()
