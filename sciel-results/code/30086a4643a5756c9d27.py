"""Improved program: greedy construction with backtracking and deadline guard."""
import time
import sys
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Greedy construction first
    values = []
    products = set()
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        # Compute all block products ending at x
        tails = []
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        tails.append(x)
        
        # Check if any product already exists
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best = list(values)
    best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    # Try to improve by inserting skipped numbers
    # Find numbers that were skipped
    used = set(values)
    skipped = [x for x in range(2, 129) if x not in used]
    
    # Try inserting each skipped number at each possible position
    # and see if we can maintain the property
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best)
        
        for pos in range(len(current) + 1):
            if time.time() > deadline:
                break
            
            # Get the set of all block products for current sequence
            curr_products = _all_block_products(current)
            
            for x in skipped:
                if time.time() > deadline:
                    break
                
                # Try inserting x at position pos
                new_seq = current[:pos] + [x] + current[pos:]
                
                # Check if this is valid
                if _is_valid(new_seq):
                    # Count how many new products we get
                    new_products = _all_block_products(new_seq)
                    # Only accept if it's a proper improvement (more elements)
                    if len(new_seq) > len(best):
                        best = new_seq
                        improved = True
                        # Remove x from skipped
                        if x in skipped:
                            skipped.remove(x)
                        break
            if improved:
                break
    
    return best


def _all_block_products(seq):
    """Get all nonempty consecutive block products."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _is_valid(seq):
    """Check if all nonempty consecutive block products are distinct."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True
