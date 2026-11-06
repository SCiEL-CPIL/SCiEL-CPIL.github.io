import time
import random
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    best_set = []
    best_size = 0
    
    def greedy_greedy(order):
        """Greedy: add x if it doesn't make 200 reachable. Uses bitmask for speed."""
        # Use a Python int as a bitset for reachable sums 0..200
        # bit s is set if sum s is reachable
        reachable = 1  # only sum 0 reachable
        selected = []
        for x in order:
            # Check: is (200 - x) reachable?
            if x > 200:
                continue
            if (reachable >> (200 - x)) & 1:
                continue
            # Safe to add
            selected.append(x)
            # Update: new_reachable = reachable | (reachable << x), masked to 200 bits
            reachable |= (reachable << x)
            reachable &= (1 << 201) - 1  # mask to sums 0..200
        
        return selected

    def local_improve(selected, skipped):
        """Try to add back skipped elements by removing one conflicting element."""
        selected_set = set(selected)
        # Rebuild reachable from selected
        reachable = 1
        for x in selected:
            reachable |= (reachable << x)
            reachable &= (1 << 201) - 1
        
        improved = list(selected)
        improved_set = set(selected)
        
        # Try each skipped element
        for x in skipped:
            if x > 200:
                continue
            if x in improved_set:
                continue
            # Check if x can be added
            if (reachable >> (200 - x)) & 1:
                continue
            # Try to add x
            improved.append(x)
            improved_set.add(x)
            # Update reachable
            new_reachable = reachable | (reachable << x)
            new_reachable &= (1 << 201) - 1
            # Check if 200 became reachable (shouldn't have, but verify)
            if (new_reachable >> 200) & 1:
                # Should not happen, but revert
                improved.pop()
                improved_set.discard(x)
                continue
            reachable = new_reachable
        
        return improved

    # Heuristic orderings
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
        [x for x in range(100, 0, -1) if x % 2 == 1] + [x for x in range(100, 0, -1) if x % 2 == 0],  # odd then even
        [x for x in range(100, 0, -1) if x > 50] + [x for x in range(50, 0, -1)],  # large first
    ]
    
    for order in orders:
        current_set = greedy_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Local improvement on best found
    if best_set:
        all_elements = set(range(1, 101))
        skipped = list(all_elements - set(best_set))
        improved = local_improve(best_set, skipped)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Randomized restarts with biased orders
    while time.time() < deadline:
        # Biased random: larger elements have higher priority in the order
        # Use exponential decay bias
        base_order = list(range(1, 101))
        random.shuffle(base_order)
        
        # Create biased order: sort by random key that favors larger values
        # Key = -random.random() * x^0.5, so larger x tends to come first
        biased_order = sorted(base_order, key=lambda x: -random.random() * (x ** 0.3))
        
        current_set = greedy_greedy(biased_order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # Try local improvement
            all_elements = set(range(1, 101))
            skipped = list(all_elements - set(best_set))
            improved = local_improve(best_set, skipped)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
        
        # Also try pure random shuffle
        order2 = list(range(1, 101))
        random.shuffle(order2)
        current_set2 = greedy_greedy(order2)
        if len(current_set2) > best_size:
            best_size = len(current_set2)
            best_set = current_set2
            all_elements = set(range(1, 101))
            skipped = list(all_elements - set(best_set))
            improved = local_improve(best_set, skipped)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    return best_set
