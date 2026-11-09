"""
Simulated Annealing construction of dense increasing sequences with unique block products.
"""
import time
import random
import math

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    def get_all_prods(seq):
        """Get set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    def is_valid(seq):
        """Check if seq is strictly increasing in [2,128] and all consecutive block products are distinct."""
        if not seq:
            return True
        if len(seq) != len(set(seq)):
            return False
        if seq != sorted(seq):
            return False
        if min(seq) < 2 or max(seq) > 128:
            return False
        n = len(seq)
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def calculate_metrics(seq):
        """
        Returns (is_valid, length, penalty)
        Penalty is a measure of invalidity (dups, inversions, range violations).
        """
        if not seq:
            return True, 0, 0
            
        n = len(seq)
        
        # Range and Uniqueness
        if any(x < 2 or x > 128 for x in seq):
            return False, n, 100
        if len(seq) != len(set(seq)):
            return False, n, 100
            
        # Order
        inversions = 0
        for i in range(n):
            for j in range(i+1, n):
                if seq[i] > seq[j]:
                    inversions += 1
        if inversions > 0:
            return False, n, inversions
            
        # Products
        products = {}
        dups = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    dups += 1
                else:
                    products[prod] = 1
                    
        if dups > 0:
            return False, n, dups
        else:
            return True, n, 0

    def score(seq):
        """
        Score for SA.
        Valid: 10000 + len
        Invalid: len - 5 * penalty
        This allows invalid states to have higher scores than short valid states,
        enabling exploration of longer (but currently invalid) sequences.
        """
        valid, length, penalty = calculate_metrics(seq)
        if valid:
            return 10000 + length
        else:
            return length - 5 * penalty

    # 1. Generate a good initial solution using a fast greedy
    def greedy_init():
        seq = []
        prods = set()
        for x in range(2, 129):
            # Check if adding x is valid
            new_prods = [x]
            prod = 1
            for i in range(len(seq) - 1, -1, -1):
                prod *= seq[i]
                new_prods.append(prod * x)
            
            # Check conflicts
            valid = True
            for p in new_prods:
                if p in prods:
                    valid = False
                    break
            if len(new_prods) != len(set(new_prods)):
                valid = False
                
            if valid:
                seq.append(x)
                prods.update(new_prods)
        
        return seq

    init_seq = greedy_init()
    if not init_seq:
        init_seq = [2, 3, 5]
        
    # Ensure init is valid
    if not is_valid(init_seq):
        init_seq = [2, 3, 5]
        
    current_seq = init_seq[:]
    current_score = score(current_seq)
    
    best_seq = current_seq[:]
    best_len = len(current_seq)
    
    # Simulated Annealing Parameters
    start_temp = 50.0
    end_temp = 0.001
    cooling_rate = 0.998 # Very slow cooling
    
    max_steps = 1000000
    t = start_temp
    
    random.seed(42)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        move_type = random.randint(1, 4)
        new_seq = current_seq[:]
        
        if move_type == 1 and len(new_seq) > 1:
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            
        elif move_type == 2 and len(new_seq) > 0:
            i = random.randint(0, len(new_seq) - 1)
            new_val = random.randint(2, 128)
            new_seq[i] = new_val
            
        elif move_type == 3:
            new_val = random.randint(2, 128)
            pos = random.randint(0, len(new_seq))
            new_seq.insert(pos, new_val)
            
        elif move_type == 4 and len(new_seq) > 1:
            i = random.randint(0, len(new_seq) - 1)
            new_seq.pop(i)
        
        # Calculate new score
        new_score = score(new_seq)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            current_seq = new_seq
            current_score = new_score
            # Update best if valid
            valid, length, _ = calculate_metrics(new_seq)
            if valid and length > best_len:
                best_len = length
                best_seq = new_seq[:]
        else:
            # Avoid division by zero or underflow
            if t > 0:
                prob = math.exp(delta / t)
                if random.random() < prob:
                    current_seq = new_seq
                    current_score = new_score
                    valid, length, _ = calculate_metrics(new_seq)
                    if valid and length > best_len:
                        best_len = length
                        best_seq = new_seq[:]
        
        # Cool down
        t *= cooling_rate
        if t < end_temp:
            t = end_temp
    
    # Final check
    if not is_valid(best_seq):
        best_seq = init_seq
        
    return best_seq
