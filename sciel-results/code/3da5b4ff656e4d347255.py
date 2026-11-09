"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm, gcd
    from itertools import combinations
    
    n = 256
    numbers = list(range(2, n + 1))
    num_count = len(numbers)
    
    # Build incompatibility: conflict[i][j] = True if lcm(numbers[i], numbers[j]) <= 256
    # Use adjacency list for efficiency
    conflict = [set() for _ in range(num_count)]
    for i in range(num_count):
        for j in range(i + 1, num_count):
            a, b = numbers[i], numbers[j]
            if lcm(a, b) <= n:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Weights: 1/x
    weights = [1.0 / numbers[i] for i in range(num_count)]
    
    # Greedy: sort by weight descending (i.e., number ascending)
    order = sorted(range(num_count), key=lambda i: -weights[i])
    
    selected = set()
    for i in order:
        # Check if i conflicts with any selected
        if not (conflict[i] & selected):
            selected.add(i)
    
    # Local search: try to improve
    def total_weight(sel):
        return sum(weights[i] for i in sel)
    
    current_weight = total_weight(selected)
    
    improved = True
    while improved:
        improved = False
        
        # Try adding a single unselected vertex
        for i in range(num_count):
            if i not in selected:
                if not (conflict[i] & selected):
                    new_sel = selected | {i}
                    new_w = current_weight + weights[i]
                    if new_w > current_weight + 1e-15:
                        selected = new_sel
                        current_weight = new_w
                        improved = True
        
        # Try swap: remove one selected, add one or more unselected
        # Try removing each selected vertex and seeing if we can add better ones
        for i in list(selected):
            temp_selected = selected - {i}
            # Try adding any unselected vertex that doesn't conflict with temp_selected
            best_gain = 0
            best_add = None
            for j in range(num_count):
                if j not in temp_selected:
                    if not (conflict[j] & temp_selected):
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = j
            if best_add is not None and best_gain > 1e-15:
                selected = temp_selected | {best_add}
                current_weight += best_gain
                improved = True
    
    # Convert back to actual numbers
    result = [numbers[i] for i in selected]
    result.sort()
    return result
# EVOLVE-BLOCK-END
