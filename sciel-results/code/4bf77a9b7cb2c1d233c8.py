import random
import numpy as np

def construct():
    n = 128
    target_min, target_max = 128, 256
    
    # Initial set: use a structured set that should cover the target range
    # Take elements spaced out to reduce clustering
    # Start with all even numbers plus some odd numbers to ensure coverage
    # Better: start with a random valid set
    
    # Let's start with a set that's known to work: 
    # If we take A = [0, 128], it works but has high multiplicity
    # We want a smaller/sparser set
    
    # Strategy: Use a set with roughly 40-50 elements
    # Start with a random subset and ensure coverage, then optimize
    
    random.seed(42)
    
    # Initial candidate: take numbers in a pattern
    # Use a set that has good spread
    base_set = set()
    # Add elements to ensure coverage: for each t in [128,256], we need a+b=t
    # A simple valid set: A = [0, 128] but let's start sparser
    
    # Start with every 3rd number plus adjustments
    A = set(range(0, 129, 3))  # ~43 elements
    # Check if this covers [128,256]
    # A+A for this set: sums of two multiples of 3 are multiples of 3
    # So it only covers multiples of 3 in the sum range - NOT sufficient
    
    # Need a set whose sumset covers all of [128,256]
    # Let's use a set with both low and high elements
    
    # Better initial set: take a mix
    # Elements from [0,64] and [64,128] to get sums in [128,256]
    
    # Start with: all numbers in [64,128] (the original) but then optimize
    # Or start with a random set of size ~45 and repair
    
    # Let's start with the original and do local search
    A = set(range(64, 129))  # 65 elements, known to work
    
    def compute_max_rep(A_set):
        """Compute max representation count in [128,256]"""
        A_list = sorted(A_set)
        max_rep = 0
        for t in range(128, 257):
            count = 0
            for i in range(len(A_list)):
                a = A_list[i]
                b = t - a
                if b < a:
                    break
                if b in A_set:
                    count += 1
            max_rep = max(max_rep, count)
        return max_rep
    
    def is_valid(A_set):
        """Check if A_set covers [128,256]"""
        A_list = sorted(A_set)
        for t in range(128, 257):
            found = False
            for i in range(len(A_list)):
                a = A_list[i]
                b = t - a
                if b < a:
                    break
                if b in A_set:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Local search: try swapping elements to reduce max rep
    current_max = compute_max_rep(A)
    best_A = set(A)
    best_max = current_max
    
    for iteration in range(2000):
        # Try removing an element and adding another
        # Focus on elements that contribute to high-representation targets
        
        # Pick a random element to remove
        if not A:
            break
        a_remove = random.choice(list(A))
        
        # Try adding a random element from [0,128] not in A
        candidates = [x for x in range(0, 129) if x not in A]
        if not candidates:
            break
        a_add = random.choice(candidates)
        
        new_A = set(A)
        new_A.remove(a_remove)
        new_A.add(a_add)
        
        if not is_valid(new_A):
            continue
        
        new_max = compute_max_rep(new_A)
        if new_max < best_max:
            best_max = new_max
            best_A = set(new_A)
            A = set(new_A)
        elif new_max == best_max and random.random() < 0.1:
            # Accept equal with some probability for diversity
            A = set(new_A)
    
    return sorted(best_A)
