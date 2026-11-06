"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import log

# EVOLVE-BLOCK-START

def _all_products(seq):
    """Return set of all nonempty consecutive-block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check strictly increasing and all block products distinct."""
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    return len(_all_products(seq)) == len(seq) * (len(seq) + 1) // 2

def _greedy_build(order):
    """Build a sequence by greedily adding elements in the given order."""
    values = []
    products = set()
    for x in order:
        if x < 2 or x > 128:
            continue
        # Check strictly increasing
        if values and x <= values[-1]:
            continue
        # Compute all new tail products
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _local_search(seq, deadline):
    """Try single-element swaps/removals/insertions to improve length."""
    best = list(seq)
    best_len = len(best)
    while time.time() < deadline:
        improved = False
        n = len(best)
        # Try removing an element and reinserting it elsewhere
        for i in range(n):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            # Try inserting best[i] at various positions
            val = best[i]
            for pos in range(len(candidate) + 1):
                if candidate and (pos > 0 and val <= candidate[pos-1] or pos < len(candidate) and val >= candidate[pos]):
                    continue
                trial = candidate[:pos] + [val] + candidate[pos:]
                if len(trial) > best_len and _is_valid(trial):
                    best = trial
                    best_len = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def _sa_search(seq, deadline):
    """Simulated annealing: try swaps of elements to improve length."""
    best = list(seq)
    best_len = len(best)
    current = list(best)
    T = 10.0
    cool = 0.999
    while time.time() < deadline and T > 0.01:
        n = len(current)
        # Pick two random positions to swap
        if n < 2:
            break
        i, j = random.sample(range(n), 2)
        if current[i] == current[j]:
            T *= cool
            continue
        trial = current[:]
        trial[i], trial[j] = trial[j], trial[i]
        # Check strictly increasing
        valid_inc = all(trial[k] < trial[k+1] for k in range(len(trial)-1))
        if valid_inc and _is_valid(trial):
            current = trial
            if len(current) > best_len:
                best = current
                best_len = len(current)
        T *= cool
    return best

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # Phase 1: Greedy initialization with natural order
    natural = list(range(2, 129))
    seq1 = _greedy_build(natural)
    best = seq1
    best_len = len(best)

    # Phase 2: Try greedy with different orderings
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        order = list(range(2, 129))
        random.shuffle(order)
        # Bias towards smaller numbers first for better packing
        seq = _greedy_build(order)
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)

    # Phase 3: Local search polish
    best = _local_search(best, deadline - 5.0)
    best_len = len(best)

    # Phase 4: Simulated annealing for further improvement
    if time.time() < deadline - 2.0:
        best = _sa_search(best, deadline - 1.0)
        best_len = len(best)

    # Final validation
    if not _is_valid(best):
        # Fallback to greedy
        best = _greedy_build(natural)

    return best

# EVOLVE-BLOCK-END
