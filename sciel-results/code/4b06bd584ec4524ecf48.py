"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def _count_pairs(family):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if len(family[i]) < len(family[j]):
                if set(family[i]).issubset(family[j]):
                    count += 1
            elif len(family[i]) > len(family[j]):
                if set(family[j]).issubset(family[i]):
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = P['n']
    m = P['m']
    
    # Initial solution: use a chain-based construction
    # Start with sets of varying sizes to create subset relationships
    random.seed(42)
    
    # Create initial family with diverse sizes
    family = []
    # Use a base set and create subsets/supersets
    base = list(range(n))
    random.shuffle(base)
    
    # Create sets of sizes 0 through 11, picking 40 distinct sets
    # Strategy: use nested chains
    # Chain 1: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11} (12 sets)
    # Then add more sets that are subsets of larger ones
    # 
    # Better: use a greedy approach to build a dense family
    
    # Initialize with a chain of length 12
    chain = [set(base[:k]) for k in range(n+1)]  # 13 sets
    family = [list(s) for s in chain]
    
    # Add more sets to reach 40
    # Add sets that are subsets of existing larger sets
    all_sets = set(frozenset(f) for f in family)
    
    # Try to add sets that create many subset relationships
    for _ in range(m - len(family)):
        best_set = None
        best_score = -1
        for _ in range(200):
            # Random subset
            size = random.randint(0, n)
            s = set(random.sample(base, size))
            if frozenset(s) in all_sets:
                continue
            # Score: how many existing sets it's a subset of + how many it contains
            score = 0
            for f in family:
                if len(s) < len(f) and s.issubset(set(f)):
                    score += 1
                elif len(s) > len(f) and set(f).issubset(s):
                    score += 1
            if score > best_score:
                best_score = score
                best_set = s
        if best_set is not None:
            family.append(list(best_set))
            all_sets.add(frozenset(best_set))
    
    current_score = _count_pairs(family)
    best_family = [list(f) for f in family]
    best_score = current_score
    
    # Local search: try swapping elements or replacing sets
    while time.time() < deadline:
        improved = False
        # Try replacing random sets with better ones
        for _ in range(500):
            if time.time() >= deadline:
                break
            idx = random.randint(0, m-1)
            old_set = set(family[idx])
            
            # Generate candidate
            size = random.randint(0, n)
            new_set = set(random.sample(base, size))
            if new_set == old_set:
                continue
            if frozenset(new_set) in {frozenset(f) for f in family if f != family[idx]}:
                continue
            
            # Evaluate
            family[idx] = list(new_set)
            new_score = _count_pairs(family)
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family = [list(f) for f in family]
            else:
                family[idx] = list(old_set)
        
        if not improved:
            # Try a different strategy: rebuild with different seed
            break
    
    return [list(f) for f in best_family]
# EVOLVE-BLOCK-END
