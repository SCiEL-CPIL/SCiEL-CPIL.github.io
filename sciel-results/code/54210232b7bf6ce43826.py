import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # We want to find a subset of denominators [2,3,...,48] such that
    # the sum of their reciprocals is strictly less than 1, and the deficit
    # (1 - sum) is minimized.
    
    # This is equivalent to finding a subset of weights w_i = 1/d_i
    # that maximizes the sum subject to sum < 1.
    
    # Since n=47, a brute-force search is infeasible.
    # However, we can use a heuristic approach:
    # 1. Start with an empty set.
    # 2. Greedily add the largest possible reciprocal that keeps the sum < 1.
    # 3. Try local improvements (swap, add, remove) to refine the solution.
    
    # A better approach:
    # Use a meet-in-the-middle strategy but with pruning.
    # Or, since the denominators are small, we can use dynamic programming
    # on the sum if we scale it appropriately.
    
    # Let's try a greedy + local search approach first.
    
    # Greedy: sort by reciprocal descending (i.e., denominator ascending)
    # and add if it fits.
    
    # But this is not optimal. Let's try a more systematic search.
    
    # Given the time constraint, let's use a randomized local search
    # with a greedy initialization.
    
    import random
    random.seed(42)
    
    def greedy_initial():
        selected = []
        current_sum = Fraction(0)
        # Sort by reciprocal descending
        for d in sorted(denominators):
            if current_sum + Fraction(1, d) < 1:
                selected.append(denominators.index(d))
                current_sum += Fraction(1, d)
        return selected, current_sum
    
    def local_search(selected, current_sum, time_limit):
        # Try to improve by swapping, adding, or removing elements
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding an element
            for i in range(n):
                if i not in selected:
                    new_sum = current_sum + Fraction(1, denominators[i])
                    if new_sum < 1:
                        # Check if this improves the deficit
                        # We want to minimize 1 - sum, so maximize sum
                        # So any valid addition that increases sum is good
                        selected.append(i)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                continue
            # Try swapping: remove one, add another
            for i in range(len(selected)):
                for j in range(n):
                    if j not in selected:
                        new_sum = current_sum - Fraction(1, denominators[selected[i]]) + Fraction(1, denominators[j])
                        if new_sum < 1:
                            # Check if this improves the deficit
                            # We want to maximize sum, so check if new_sum > current_sum
                            if new_sum > current_sum:
                                selected[i] = j
                                current_sum = new_sum
                                improved = True
                                break
                if improved:
                    break
            if improved:
                continue
            # Try removing an element (only if it doesn't reduce sum, which it always does)
            # So removing is never beneficial for maximizing sum.
            # But we might need to remove to allow adding a larger element.
            # This is covered by the swap.
        
        return selected, current_sum
    
    # Initialize with greedy
    selected, current_sum = greedy_initial()
    
    # Local search
    selected, current_sum = local_search(selected, current_sum, deadline - 5)
    
    # Convert to indices
    # selected already contains indices
    
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        return selected
    else:
        # Fallback
        return [0]
