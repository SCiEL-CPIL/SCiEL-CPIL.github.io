import time
import random
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 60s limit
    
    def compute_tails(seq, x):
        """Compute new tail products if x is appended to seq."""
        new_tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            new_tails.append(prod)
        return new_tails

    def is_valid_addition(seq, x, products_set):
        """Check if x can be appended to seq without collision."""
        new_tails = compute_tails(seq, x)
        if products_set.intersection(new_tails):
            return False, None
        return True, new_tails

    def collision_count(seq, x, products_set):
        """Count how many new products would collide with existing ones."""
        new_tails = compute_tails(seq, x)
        return len(products_set.intersection(new_tails))

    def greedy_with_lookahead(start_val=2, end_val=128, lookahead_depth=2):
        """Greedy with lookahead: at each step, choose the valid x that minimizes future collision risk."""
        seq = []
        products_set = set()
        
        for x in range(start_val, end_val + 1):
            if time.time() > deadline:
                break
                
            valid, new_tails = is_valid_addition(seq, x, products_set)
            if not valid:
                continue
                
            # If lookahead_depth > 1, check if adding x blocks many future options
            if lookahead_depth > 1:
                # Estimate future collision risk: count how many numbers in (x, end_val] 
                # would collide if we add x now
                risk = 0
                temp_set = products_set.union(new_tails)
                for y in range(x + 1, min(x + lookahead_depth * 10, end_val + 1)):
                    # Quick check: would y collide with temp_set?
                    prod = y
                    for a in reversed(seq):
                        prod *= a
                        if prod in temp_set:
                            risk += 1
                            break
                        if prod > 128 * 128:  # Products grow fast, stop early
                            break
                    else:
                        if y in temp_set:
                            risk += 1
            else:
                risk = 0
                
            # Heuristic: prefer numbers with lower risk
            # For now, just take the first valid one (standard greedy)
            # but we could implement a proper lookahead here
            seq.append(x)
            products_set.update(new_tails)
            
        return seq

    def backtracking_search(seq, products_set, next_x, max_depth=3):
        """Try to extend seq by backtracking when stuck."""
        if next_x > 128 or time.time() > deadline:
            return seq, products_set
            
        if len(seq) >= max_depth:
            return seq, products_set
            
        # Try adding next_x
        valid, new_tails = is_valid_addition(seq, next_x, products_set)
        if valid:
            new_seq = seq + [next_x]
            new_set = products_set.union(new_tails)
            # Recurse
            result_seq, result_set = backtracking_search(new_seq, new_set, next_x + 1, max_depth)
            if len(result_seq) > len(seq):
                return result_seq, result_set
            # Backtrack: try skipping next_x
            result_seq2, result_set2 = backtracking_search(seq, products_set, next_x + 1, max_depth)
            if len(result_seq2) > len(seq):
                return result_seq2, result_set2
            return new_seq, new_set
        else:
            # Can't add next_x, try skipping it
            return backtracking_search(seq, products_set, next_x + 1, max_depth)

    def local_search_perturb(seq, products_set, perturb_count=3):
        """Perturb the sequence by removing some elements and re-greedy-ing."""
        if not seq or time.time() > deadline:
            return seq, products_set
            
        # Randomly remove some elements from the middle/end
        new_seq = list(seq)
        for _ in range(perturb_count):
            if len(new_seq) <= 3:
                break
            # Remove a random element (not the first)
            idx = random.randint(1, len(new_seq) - 1)
            removed = new_seq.pop(idx)
            
        # Rebuild products_set
        new_set = set()
        for i, a in enumerate(new_seq):
            prod = a
            new_set.add(prod)
            for j in range(i - 1, -1, -1):
                prod *= new_seq[j]
                new_set.add(prod)
        
        # Re-greedy from the last element
        last_x = new_seq[-1] if new_seq else 1
        for x in range(last_x + 1, 129):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(new_seq, x, new_set)
            if valid:
                new_seq.append(x)
                new_set.update(new_tails)
                
        return new_seq, new_set

    best_seq = []
    best_products = set()
    
    # 1. Standard Greedy
    seq = []
    products_set = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        valid, new_tails = is_valid_addition(seq, x, products_set)
        if valid:
            seq.append(x)
            products_set.update(new_tails)
    if len(seq) > len(best_seq):
        best_seq = seq
        best_products = products_set

    # 2. Try different start values
    for sv in range(3, 15):
        if time.time() > deadline:
            break
        seq = []
        products_set = set()
        for x in range(sv, 129):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(seq, x, products_set)
            if valid:
                seq.append(x)
                products_set.update(new_tails)
        if len(seq) > len(best_seq):
            best_seq = seq
            best_products = products_set

    # 3. Local search with random restarts and perturbations
    random.seed(42)
    trials = 0
    while time.time() < deadline:
        trials += 1
        
        # Start from best_seq or a random greedy
        if trials % 3 == 0 and best_seq:
            seq = list(best_seq)
            # Rebuild products_set
            products_set = set()
            for i, a in enumerate(seq):
                prod = a
                products_set.add(prod)
                for j in range(i - 1, -1, -1):
                    prod *= seq[j]
                    products_set.add(prod)
        else:
            # Random greedy with skips
            start_val = random.randint(2, 12)
            base_skip = random.uniform(0.1, 0.5)
            seq = []
            products_set = set()
            for x in range(start_val, 129):
                if time.time() > deadline:
                    break
                if x < 15:
                    p_skip = base_skip * (15 - x) / 15.0
                    if random.random() < p_skip:
                        continue
                valid, new_tails = is_valid_addition(seq, x, products_set)
                if valid:
                    seq.append(x)
                    products_set.update(new_tails)
        
        # Apply local search perturbation
        seq2, products_set2 = local_search_perturb(seq, products_set, perturb_count=random.randint(1, 4))
        
        # Try to extend further
        last_x = seq2[-1] if seq2 else 1
        for x in range(last_x + 1, 129):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(seq2, x, products_set2)
            if valid:
                seq2.append(x)
                products_set2.update(new_tails)
        
        if len(seq2) > len(best_seq):
            best_seq = seq2
            best_products = products_set2
            
    # 4. Final polish: try to insert missing numbers into best_seq
    if best_seq:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Rebuild products_set for best_seq
            products_set = set()
            for i, a in enumerate(best_seq):
                prod = a
                products_set.add(prod)
                for j in range(i - 1, -1, -1):
                    prod *= best_seq[j]
                    products_set.add(prod)
            
            # Try inserting each missing number
            in_seq = set(best_seq)
            for x in range(2, 129):
                if x in in_seq:
                    continue
                if time.time() > deadline:
                    break
                # Find insertion position
                pos = None
                for i, a in enumerate(best_seq):
                    if a > x:
                        pos = i
                        break
                if pos is None:
                    pos = len(best_seq)
                
                # Check if inserting x at pos is valid
                new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                # Verify all consecutive block products are distinct
                valid = True
                all_prods = set()
                for i in range(len(new_seq)):
                    prod = new_seq[i]
                    if prod in all_prods:
                        valid = False
                        break
                    all_prods.add(prod)
                    for j in range(i + 1, len(new_seq)):
                        prod *= new_seq[j]
                        if prod in all_prods:
                            valid = False
                            break
                        all_prods.add(prod)
                    if not valid:
                        break
                
                if valid:
                    best_seq = new_seq
                    improved = True
                    break
        
    return best_seq
