<<<<<<< SEARCH
import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_incremental(A, old_val, new_val):
    """
    Compute new score when replacing old_val with new_val in set A.
    A is a list of 20 distinct positive integers.
    Returns the new max(|A+A|, |A*A|).
    """
    # Build sum_set and prod_set from A without old_val
    sum_set = set()
    prod_set = set()
    others = [x for x in A if x != old_val]
    
    for i in range(len(others)):
        a = others[i]
        for j in range(i, len(others)):
            b = others[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    
    # Now add contributions involving new_val
    sum_set.add(2 * new_val)
    prod_set.add(new_val * new_val)
    
    for x in others:
        sum_set.add(new_val + x)
        prod_set.add(new_val * x)
    
    return max(len(sum_set), len(prod_set))

def generate_smooth_pool(max_val=10000):
    """Generate 2,3,5-smooth numbers <= max_val."""
    smooth = set()
    for i in range(0, 14):  # 2^i
        for j in range(0, 10):  # 3^j
            for k in range(0, 7):  # 5^k
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= max_val and val >= 1:
                    smooth.add(val)
    return sorted(smooth)
=======
import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_incremental(A, old_val, new_val):
    """
    Compute new score when replacing old_val with new_val in set A.
    A is a list of 20 distinct positive integers.
    Returns the new max(|A+A|, |A*A|).
    """
    # Build sum_set and prod_set from A without old_val
    sum_set = set()
    prod_set = set()
    others = [x for x in A if x != old_val]
    
    for i in range(len(others)):
        a = others[i]
        for j in range(i, len(others)):
            b = others[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    
    # Now add contributions involving new_val
    sum_set.add(2 * new_val)
    prod_set.add(new_val * new_val)
    
    for x in others:
        sum_set.add(new_val + x)
        prod_set.add(new_val * x)
    
    return max(len(sum_set), len(prod_set))

def generate_smooth_pool(max_val=10000, primes=(2,3)):
    """Generate p-smooth numbers <= max_val for given primes."""
    smooth = set()
    if len(primes) == 2:
        p, q = primes
        for i in range(0, 15):
            pi = p ** i
            if pi > max_val:
                break
            for j in range(0, 12):
                val = pi * (q ** j)
                if val <= max_val and val >= 1:
                    smooth.add(val)
    elif len(primes) == 3:
        p, q, r = primes
        for i in range(0, 14):
            pi = p ** i
            if pi > max_val:
                break
            for j in range(0, 10):
                pij = pi * (q ** j)
                if pij > max_val:
                    break
                for k in range(0, 8):
                    val = pij * (r ** k)
                    if val <= max_val and val >= 1:
                        smooth.add(val)
    return sorted(smooth)
>>>>>>> REPLACE

