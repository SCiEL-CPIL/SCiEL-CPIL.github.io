import time
import random

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    def compute_all_products(values):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                products.add(prod)
        return products
    
    def is_valid_insertion(values, products, pos, x):
        """Check if x can be inserted at position pos in values without product collision."""
        # Check strictly increasing
        if pos > 0 and values[pos-1] >= x:
            return False
        if pos < len(values) and x >= values[pos]:
            return False
        
        # Compute all new products involving x
        # Products that end at x: x, values[pos-1]*x, values[pos-2]*values[pos-1]*x, ...
        # Products that start at x: x, x*values[pos], x*values[pos]*values[pos+1], ...
        # Products that contain x in the middle: values[i]*...*values[pos-1]*x*values[pos]*...*values[j]
        
        # Actually, it's easier to just compute all products of the new sequence
        new_values = values[:pos] + [x] + values[pos:]
        new_products = compute_all_products(new_values)
        
        # Check for duplicates
        n = len(new_values)
        total_products = sum(n - i for i in range(n))
        if len(new_products) != total_products:
            return False
        
        return True
    
    def greedy_extend(values, products, start):
        """Continue greedily from current state by appending."""
        for x in range(start, 129):
            if time.time() > deadline:
                break
            # Check if x can be appended
            new_values = values + [x]
            new_products = compute_all_products(new_values)
            n = len(new_values)
            total_products = sum(n - i for i in range(n))
            if len(new_products) == total_products:
                values = new_values
                products = new_products
        return values, products
    
    # Greedy baseline: start from 2 and greedily add smallest valid
    values = []
    products = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        new_values = values + [x]
        new_products = compute_all_products(new_values)
        n = len(new_values)
        total_products = sum(n - i for i in range(n))
        if len(new_products) == total_products:
            values = new_values
            products = new_products
    
    best = list(values)
    best_len = len(best)
    
    # Local search: try to improve by inserting new elements
    current = list(best)
    current_products = compute_all_products(current)
    current_len = len(current)
    
    random.seed(42)
    
    # Try many random insertions
    max_iterations = 50000
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Try to insert a random element at a random position
        if len(current) < 127:  # Max possible length is 127 (2 to 128)
            pos = random.randint(0, len(current))
            
            if pos == 0:
                lo, hi = 2, (current[0] - 1 if current else 128)
            elif pos == len(current):
                lo, hi = (current[-1] + 1 if current else 2), 128
            else:
                lo, hi = (current[pos-1] + 1 if pos > 0 else 2), (current[pos] - 1 if pos < len(current) else 128)
            
            if lo > hi:
                continue
            
            # Try a few random values in the range
            for _ in range(3):
                if lo > hi:
                    break
                x = random.randint(lo, hi)
                if is_valid_insertion(current, current_products, pos, x):
                    new_current = current[:pos] + [x] + current[pos:]
                    new_products = compute_all_products(new_current)
                    current = new_current
                    current_products = new_products
                    current_len = len(current)
                    
                    if current_len > best_len:
                        best = list(current)
                        best_len = current_len
                    break
    
    # Also try removing and re-adding elements
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
        
        if len(current) < 2:
            break
        
        # Remove a random element
        pos = random.randint(0, len(current) - 1)
        removed = current[pos]
        new_current = current[:pos] + current[pos+1:]
        
        if not new_current:
            continue
        
        new_products = compute_all_products(new_current)
        
        # Try to add back a different element or extend
        # First, try to greedily extend from the end
        extended = list(new_current)
        ext_products = set(new_products)
        for x in range(new_current[-1] + 1, 129):
            if time.time() > deadline - 5:
                break
            test_values = extended + [x]
            test_products = compute_all_products(test_values)
            n = len(test_values)
            total = sum(n - i for i in range(n))
            if len(test_products) == total:
                extended = test_values
                ext_products = test_products
        
        if len(extended) > current_len:
            current = extended
            current_products = ext_products
            current_len = len(current)
            
            if current_len > best_len:
                best = list(current)
                best_len = current_len
    
    return best
