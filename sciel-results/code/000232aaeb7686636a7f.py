"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import sys
    
    deadline = time.time() + 85  # Leave 5s margin before 90s kill
    
    n = 47  # denominators 2 to 48
    
    # Suffix sums for pruning: suffix_sum[i] = sum of 1/(i+2) + ... + 1/48
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, i + 2)
    
    # Strong initial solution: 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806, deficit = 1/1806
    best_sum = Fraction(1805, 1806)
    best_indices = [0, 1, 5, 41]  # den 2, 3, 7, 43
    
    sys.setrecursionlimit(10000)
    
    def dfs(idx, current_sum, selected_list):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        # Prune: even adding all remaining can't beat best
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = selected_list[:]
            return
        
        # Try including current denominator first (to find good solutions quickly)
        new_sum = current_sum + Fraction(1, idx + 2)
        if new_sum < 1:
            selected_list.append(idx)
            dfs(idx + 1, new_sum, selected_list)
            selected_list.pop()
        
        # Try excluding
        dfs(idx + 1, current_sum, selected_list)
    
    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
