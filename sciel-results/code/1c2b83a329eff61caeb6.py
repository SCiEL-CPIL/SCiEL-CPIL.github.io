def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 50
    
    denominators = list(range(2, 49))
    fractions = [Fraction(1, d) for d in denominators]
    n = len(fractions)
    
    # Split: first 20 are "large", rest are "small"
    split = 20
    large_fracs = fractions[:split]
    small_fracs = fractions[split:]
    
    best_sum = Fraction(0)
    best_indices = []
    
    def greedy_fill(current_sum, small_start_idx):
        """Greedily add small fractions to maximize sum while keeping < 1."""
        total = current_sum
        selected = []
        for i, f in enumerate(small_fracs):
            if total + f < 1:
                total += f
                selected.append(small_start_idx + i)
        return total, selected
    
    # Enumerate subsets of large fractions
    def enumerate_large(idx, current_sum, selected_large):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
        
        if idx == split:
            # Greedily fill in small fractions
            total, selected_small = greedy_fill(current_sum, split)
            if total < 1 and total > best_sum:
                best_sum = total
                best_indices = selected_large + selected_small
            return
        
        # Try including
        new_sum = current_sum + large_fracs[idx]
        if new_sum < 1:
            selected_large.append(idx)
            enumerate_large(idx + 1, new_sum, selected_large)
            selected_large.pop()
        
        # Try excluding
        enumerate_large(idx + 1, current_sum, selected_large)
    
    enumerate_large(0, Fraction(0), [])
    
    return best_indices
