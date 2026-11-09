"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _independence_number(triples, n=16):
    """Compute the largest subset of points containing no triple."""
    triple_set = set()
    for t in triples:
        triple_set.add(tuple(sorted(t)))
    
    # Try all subsets from largest to smallest
    # For n=16, we can check subsets of size k from 16 down to 1
    # But this is expensive. Use a smarter approach:
    # A subset S is independent if no triple is fully contained in S.
    
    # Start with all points
    all_points = set(range(n))
    current_size = n
    
    # Check if the full set contains a triple
    for t in triple_set:
        if all(p in all_points for p in t):
            # Full set is not independent, need to find max independent set
            break
    else:
        return n
    
    # Use a greedy/heuristic approach to find large independent sets
    # and verify with exact check for smaller sizes
    
    def is_independent(subset):
        s = set(subset)
        for t in triple_set:
            if all(p in s for p in t):
                return False
        return True
    
    # Try to find the maximum independent set using a simple approach
    # For 16 points, we can try subsets of various sizes
    best = 0
    
    # Check sizes from n-1 down to 1, stop when we find one that works
    # Actually, we want the LARGEST, so check from n down
    for size in range(n, 0, -1):
        # Check all subsets of this size
        found = False
        for subset in combinations(range(n), size):
            if is_independent(subset):
                return size
        # If we get here, no subset of this size is independent
        # so the answer is less than this size
    
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _generate_random_triples(n=16, num_triples=40, seed=42):
    """Generate random linear triples."""
    rng = random.Random(seed)
    triples = []
    attempts = 0
    max_attempts = 10000
    
    while len(triples) < num_triples and attempts < max_attempts:
        attempts += 1
        t = tuple(sorted(rng.sample(range(n), 3)))
        # Check linearity
        linear = True
        for existing in triples:
            common = set(t) & set(existing)
            if len(common) > 1:
                linear = False
                break
        if linear:
            triples.append(list(t))
    
    return triples

def _local_search(triples, n=16, max_iter=500, seed=0):
    """Improve triples via local search to minimize independence number."""
    rng = random.Random(seed)
    triples = [list(t) for t in triples]
    
    if not _is_linear(triples):
        return triples
    
    best_triples = [list(t) for t in triples]
    best_indep = _independence_number(triples, n)
    
    for _ in range(max_iter):
        # Try to add a new random triple
        t = tuple(sorted(rng.sample(range(n), 3)))
        linear = True
        for existing in triples:
            common = set(t) & set(existing)
            if len(common) > 1:
                linear = False
                break
        if linear:
            new_triples = triples + [list(t)]
            indep = _independence_number(new_triples, n)
            if indep < best_indep:
                best_indep = indep
                best_triples = [list(x) for x in new_triples]
                triples = new_triples
        else:
            # Try to replace a random triple
            if triples:
                idx = rng.randint(0, len(triples)-1)
                old = triples[idx]
                # Try several replacements
                for _ in range(10):
                    t = tuple(sorted(rng.sample(range(n), 3)))
                    # Check linearity with all except the one being replaced
                    linear = True
                    for j, existing in enumerate(triples):
                        if j == idx:
                            continue
                        common = set(t) & set(existing)
                        if len(common) > 1:
                            linear = False
                            break
                    if linear:
                        new_triples = triples[:idx] + [list(t)] + triples[idx+1:]
                        indep = _independence_number(new_triples, n)
                        if indep < best_indep:
                            best_indep = indep
                            best_triples = [list(x) for x in new_triples]
                            triples = new_triples
                        break
    
    return best_triples

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    best_triples = None
    best_indep = 16
    
    # Start with a good initial set
    initial = _generate_random_triples(16, 50, seed=42)
    if _is_linear(initial):
        best_triples = initial
        best_indep = _independence_number(initial, 16)
    
    # Try multiple seeds and local search
    for seed in range(20):
        if time.time() > deadline:
            break
        
        triples = _generate_random_triples(16, 50, seed=seed)
        if not _is_linear(triples):
            continue
        
        improved = _local_search(triples, 16, max_iter=200, seed=seed)
        indep = _independence_number(improved, 16)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    if best_triples is None:
        # Fallback to simple valid system
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
