"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import random

N = 512

def _precompute():
    """Precompute for each x the list of (a, b) pairs such that {a,b,c,x} is a 4-term AP."""
    # For each x, find all triples (a, b, c) distinct from x and each other
    # such that {a,b,c,x} forms a 4-term AP.
    # A 4-term AP: p, p+d, p+2d, p+3d for d>0.
    # For a given x, x can be at position 0,1,2,3.
    pairs_for_x = [[] for _ in range(N + 1)]  # 1-indexed
    
    for x in range(1, N + 1):
        seen_pairs = set()
        # x is the 4th term: x = p + 3d => p = x - 3d, terms: p, p+d, p+2d, x
        for d in range(1, (x - 1) // 3 + 1):
            p = x - 3 * d
            if p < 1:
                continue
            t1, t2, t3 = p, p + d, p + 2 * d
            # The other three terms are t1, t2, t3
            # Any pair among {t1,t2,t3} combined with x forms the AP
            # We need: if both elements of a pair are in S, adding x is forbidden
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = min(t1, t2, t3)[i] if False else sorted([t1, t2, t3])[i], sorted([t1, t2, t3])[j]
                    # Just use the actual values
                    pass
            # Let me just collect all pairs from {t1, t2, t3}
            terms = [t1, t2, t3]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = terms[i], terms[j]
                    if a >= 1 and a <= N and b >= 1 and b <= N:
                        key = (min(a, b), max(a, b))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            pairs_for_x[x].append(key)
        
        # x is the 3rd term: x = p + 2d => p = x - 2d, terms: p, p+d, x, x+d
        for d in range(1, (x - 1) // 2 + 1):
            p = x - 2 * d
            if p < 1:
                continue
            t4 = x + d
            if t4 > N:
                continue
            terms = [p, p + d, t4]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = terms[i], terms[j]
                    if 1 <= a <= N and 1 <= b <= N:
                        key = (min(a, b), max(a, b))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            pairs_for_x[x].append(key)
        
        # x is the 2nd term: x = p + d => p = x - d, terms: p, x, x+d, x+2d
        for d in range(1, N - x + 1):
            p = x - d
            if p < 1:
                continue
            t3 = x + d
            t4 = x + 2 * d
            if t4 > N:
                continue
            terms = [p, t3, t4]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = terms[i], terms[j]
                    if 1 <= a <= N and 1 <= b <= N:
                        key = (min(a, b), max(a, b))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            pairs_for_x[x].append(key)
        
        # x is the 1st term: x = p, terms: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            t2 = x + d
            t3 = x + 2 * d
            t4 = x + 3 * d
            if t4 > N:
                continue
            terms = [t2, t3, t4]
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = terms[i], terms[j]
                    if 1 <= a <= N and 1 <= b <= N:
                        key = (min(a, b), max(a, b))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            pairs_for_x[x].append(key)
    
    return pairs_for_x

def construct():
    random.seed(42)
    pairs_for_x = _precompute()
    
    # Greedy initialization: add elements in random order if valid
    current = set()
    current_mask = 0  # bitmask of current set
    
    def can_add(x):
        """Check if x can be added to current set without forming a 4-term AP."""
        for (a, b) in pairs_for_x[x]:
            if a in current and b in current:
                return False
        return True
    
    # Greedy seed
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        if can_add(x):
            current.add(x)
            current_mask |= (1 << x)
    
    best_set = set(current)
    best_size = len(current)
    
    # Simulated annealing
    T = 50.0
    cooling = 0.999
    iterations = 200000
    
    for it in range(iterations):
        if T < 0.01:
            break
        
        # Random move: try to add a random element or remove a random element
        if random.random() < 0.5 and len(current) < N:
            # Try to add a random element
            x = random.randint(1, N)
            if x not in current and can_add(x):
                current.add(x)
                current_mask |= (1 << x)
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            else:
                # Worsening move: add anyway? No, we can't add if it creates AP.
                # Instead, try a swap: remove one and add another
                pass
        else:
            # Try to remove a random element and add another
            if current:
                x_remove = random.choice(list(current))
                # Find a candidate to add
                candidates = [x for x in range(1, N + 1) if x not in current]
                if candidates:
                    x_add = random.choice(candidates)
                    # Temporarily remove x_remove
                    current.discard(x_remove)
                    if can_add(x_add):
                        current.add(x_add)
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = set(current)
                    else:
                        # Revert
                        current.add(x_remove)
        
        T *= cooling
    
    # Return the best set found
    return sorted(best_set)
# EVOLVE-BLOCK-END
