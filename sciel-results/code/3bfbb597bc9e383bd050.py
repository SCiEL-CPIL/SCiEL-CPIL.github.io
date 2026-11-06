import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    MASK = (1 << 201) - 1
    
    def is_valid(bits):
        return not (bits & (1 << 200))
        
    def add_to_bits(bits, x):
        return (bits | (bits << x)) & MASK

    def greedy_construct(order):
        bits = 1  # bit 0 is set
        selected = []
        for x in order:
            new_bits = add_to_bits(bits, x)
            if is_valid(new_bits):
                bits = new_bits
                selected.append(x)
        return selected, bits

    def local_search_add(selected, bits):
        """Try to add back excluded elements greedily."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in excluded[:]:
                if time.time() > deadline:
                    break
                new_bits = add_to_bits(bits, x)
                if is_valid(new_bits):
                    selected.append(x)
                    selected_set.add(x)
                    bits = new_bits
                    excluded.remove(x)
                    improved = True
        return selected, bits

    def local_search_swap(selected, bits):
        """Try 1-for-1 swaps to enable further additions."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each selected element and see if we can add a different one
            for i, x in enumerate(selected[:]):
                if time.time() > deadline:
                    break
                # Recalculate bits without x
                temp_bits = 1
                for j, y in enumerate(selected):
                    if j != i:
                        temp_bits = add_to_bits(temp_bits, y)
                
                # Try adding each excluded element
                for y in excluded[:]:
                    if time.time() > deadline:
                        break
                    new_bits = add_to_bits(temp_bits, y)
                    if is_valid(new_bits):
                        # Perform the swap
                        selected[i] = y
                        selected_set.remove(x)
                        selected_set.add(y)
                        excluded.remove(y)
                        excluded.append(x)
                        bits = new_bits
                        improved = True
                        break  # Restart the outer loop with new state
                if improved:
                    break
        
        # After swaps, try to add more elements
        selected, bits = local_search_add(selected, bits)
        return selected, bits

    # Generate initial candidates with diverse orders
    best_selected = []
    best_bits = 1
    best_size = 0
    
    orders = [
        list(range(100, 0, -1)),  # descending
        list(range(1, 101)),      # ascending
    ]
    
    # Add a "balanced" order: alternate high and low
    balanced = []
    for i in range(50):
        balanced.append(100 - i)  # 100, 99, ..., 51
        if i < 50:
            balanced.append(i + 1)  # 1, 2, ..., 50
    orders.append(balanced)
    
    rng = random.Random(42)
    for _ in range(150):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if time.time() > deadline:
            break
        sel, bits = greedy_construct(order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_bits = bits
            
    # Local search on the best candidate
    if best_selected:
        # First pass: Greedy Add
        best_selected, best_bits = local_search_add(best_selected, best_bits)
        
        # Second pass: Swaps (1-for-1 to enable more additions)
        best_selected, best_bits = local_search_swap(best_selected, best_bits)
        
        # Third pass: Greedy Add again
        best_selected, best_bits = local_search_add(best_selected, best_bits)
        
        # Fourth pass: More swaps if time allows
        if time.time() < deadline:
            best_selected, best_bits = local_search_swap(best_selected, best_bits)
            best_selected, best_bits = local_search_add(best_selected, best_bits)

    # Fallback
    if not best_selected:
        best_selected = [100]
        
    return sorted(best_selected)
