def construct():
    from fractions import Fraction
    import random
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def sum_of(indices):
        return sum(Fraction(1, denominators[i]) for i in indices)
    
    # Greedy initialization
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + Fraction(1, denominators[i]) < 1:
            selected.add(i)
            total += Fraction(1, denominators[i])
    
    best_sum = total
    best_indices = list(selected)
    
    # Local search: try to improve by adding, removing, or swapping
    def try_improve(current_set, current_sum):
        nonlocal best_sum, best_indices
        
        improved = True
        while improved:
            improved = False
            
            # Try adding each element not in set
            for i in range(n):
                if i not in current_set:
                    new_sum = current_sum + Fraction(1, denominators[i])
                    if new_sum < 1 and new_sum > best_sum:
                        current_set.add(i)
                        current_sum = new_sum
                        best_sum = new_sum
                        best_indices = list(current_set)
                        improved = True
            
            # Try removing each element in set (if it allows adding a better one)
            # This is more complex, let's try swaps
            
            # Try swapping: remove one, add another
            for i in list(current_set):
                sum_without_i = current_sum - Fraction(1, denominators[i])
                for j in range(n):
                    if j not in current_set:
                        new_sum = sum_without_i + Fraction(1, denominators[j])
                        if new_sum < 1 and new_sum > best_sum:
                            current_set.remove(i)
                            current_set.add(j)
                            current_sum = new_sum
                            best_sum = new_sum
                            best_indices = list(current_set)
                            improved = True
                            break
                if improved:
                    break
        
        return current_set, current_sum
    
    selected, total = try_improve(selected, total)
    
    return best_indices
