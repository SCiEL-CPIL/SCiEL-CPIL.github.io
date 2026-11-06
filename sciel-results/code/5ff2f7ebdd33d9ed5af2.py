# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _find_bad_triple(selected):
    """Find a bad triple in selected. Returns (i,j,k) indices or None."""
    n = len(selected)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = gcd(selected[i], selected[j])
            for k in range(j+1, n):
                d_ik = gcd(selected[i], selected[k])
                if d_ik != d_ij:
                    continue
                d_jk = gcd(selected[j], selected[k])
                if d_jk == d_ij:
                    return (i, j, k)
    return None

def _count_bad_triples(selected):
    """Count bad triples and return count per element."""
    n = len(selected)
    counts = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            d_ij = gcd(selected[i], selected[j])
            for k in range(j+1, n):
                d_ik = gcd(selected[i], selected[k])
                if d_ik != d_ij:
                    continue
                d_jk = gcd(selected[j], selected[k])
                if d_jk == d_ij:
                    counts[i] += 1
                    counts[j] += 1
                    counts[k] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_set = set()
    
    # Strategy 1: Greedy removal from full set
    selected = list(range(1, 129))
    
    while time.time() < deadline:
        # Find all bad triples and count participation
        counts = _count_bad_triples(selected)
        max_count = max(counts)
        if max_count == 0:
            break
        
        # Remove the element with highest bad-triple participation
        # Ties broken by smallest value (prefer keeping larger numbers)
        max_indices = [i for i, c in enumerate(counts) if c == max_count]
        # Prefer removing smaller numbers to keep larger ones
        remove_idx = min(max_indices, key=lambda i: selected[i])
        selected.pop(remove_idx)
    
    best_set = set(selected)
    
    # Strategy 2: Try adding elements back in different orders
    if time.time() < deadline - 10:
        # Try a different greedy: build up from scratch with different orderings
        for order_type in ['ascending', 'descending', 'prime_first', 'composite_first']:
            if time.time() > deadline - 5:
                break
            
            if order_type == 'ascending':
                order = list(range(1, 129))
            elif order_type == 'descending':
                order = list(range(128, 0, -1))
            elif order_type == 'prime_first':
                primes = [x for x in range(1, 129) if x > 1 and all(x % p != 0 for p in range(2, int(x**0.5)+1))]
                composites = [x for x in range(1, 129) if x not in primes]
                order = primes + composites
            else:  # composite_first
                primes = [x for x in range(1, 129) if x > 1 and all(x % p != 0 for p in range(2, int(x**0.5)+1))]
                composites = [x for x in range(1, 129) if x not in primes]
                order = composites + primes
            
            result = []
            for x in order:
                if all(len({gcd(a, b), gcd(a, x), gcd(b, x)}) > 1 for a, b in combinations(result, 2)):
                    result.append(x)
            
            if len(result) > len(best_set):
                best_set = set(result)
    
    # Strategy 3: Local search - try swapping elements
    if time.time() < deadline - 5:
        current = list(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try removing each element and see if we can add others
            for i in range(len(current)):
                if time.time() > deadline - 2:
                    break
                trial = current[:i] + current[i+1:]
                # Try to add any element from 1..128 not in trial
                for x in range(1, 129):
                    if x in trial:
                        continue
                    if all(len({gcd(a, b), gcd(a, x), gcd(b, x)}) > 1 for a, b in combinations(trial, 2)):
                        if len(trial) + 1 > len(current):
                            current = trial + [x]
                            improved = True
                            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
