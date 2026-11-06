"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # We want to maximize the sum, so we process larger fractions first.
    # This means smaller denominators first.
    # The indices in the problem are 0-based into [2, 3, ..., 48].
    # So index 0 is 1/2, index 1 is 1/3, etc.
    # This is already in descending order of value.
    
    unit_vals = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum(unit_vals[i:])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + unit_vals[i]

    best_sum = Fraction(0)
    best_indices = []
    
    # To avoid deep recursion limits or overhead, we can use an iterative stack or just recursion with sys.setrecursionlimit.
    # Depth is 47, so recursion is fine.
    
    # Global variables for the DFS
    # We need to update best_sum and best_indices from within the function.
    # Using nonlocal or a list/dict wrapper.
    state = {
        'best_sum': Fraction(0),
        'best_indices': [],
        'time_out': False
    }

    def dfs(i, current_sum, current_indices):
        nonlocal state
        
        # Time check
        if time.time() > deadline:
            state['time_out'] = True
            return

        # Pruning: If current sum + max possible remaining sum <= best sum, prune
        if current_sum + suffix_sum[i] <= state['best_sum']:
            return

        # Base case
        if i == n:
            if current_sum > state['best_sum']:
                state['best_sum'] = current_sum
                state['best_indices'] = list(current_indices)
            return

        # Branch 1: Take item i
        # Check if adding unit_vals[i] keeps sum strictly below 1
        next_sum_take = current_sum + unit_vals[i]
        if next_sum_take < 1:
            # Recurse with item i included
            current_indices.append(i)
            dfs(i + 1, next_sum_take, current_indices)
            current_indices.pop()
            
            if state['time_out']:
                return

        # Branch 2: Skip item i
        dfs(i + 1, current_sum, current_indices)

    # Initialize best with a simple greedy solution to provide a tighter bound initially
    # Greedy: pick largest available that fits
    greedy_sum = Fraction(0)
    greedy_indices = []
    for i in range(n):
        if greedy_sum + unit_vals[i] < 1:
            greedy_sum += unit_vals[i]
            greedy_indices.append(i)
            
    if greedy_sum > state['best_sum']:
        state['best_sum'] = greedy_sum
        state['best_indices'] = greedy_indices

    # Run DFS
    dfs(0, Fraction(0), [])

    # Return the best indices found
    if not state['best_indices']:
        # Fallback if no solution found (should not happen as 1/2 is valid)
        return [0]
        
    return state['best_indices']

# EVOLVE-BLOCK-END
