import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 192
    all_vals = list(range(1, N + 1))
    
    # Precompute: for each pair (i, j), i < j, the values s = i+j
    # We'll use this for fast checking
    # Actually, let's precompute for each a, which triples (b,c,d) have b+c+d ≡ 0 mod a
    # That's too expensive to precompute fully. Instead, let's use a different approach.
    
    # Key insight: For a given set S and candidate a, we need to check:
    # 1. a does not divide sum of any 3 distinct elements in S (excluding a if a is in S)
    # 2. For each b in S, b does not divide sum of any 3 distinct elements in (S ∪ {a}) \ {b}
    
    # Fast check: For a set S, and for each element x in S, 
    # we need: no 3 distinct elements y,z,w in S\{x} have y+z+w ≡ 0 (mod x)
    
    # Let's use a bitset-based approach for the triple sum check.
    # For a given modulus m, and a set of numbers, we want to know if there exist
    # 3 distinct elements whose sum is divisible by m.
    
    # Alternative: Precompute for each a in [1,192], and for each pair (b,c) with b<c,
    # the residue (b+c) mod a. Then for a third element d, we check if (b+c+d) mod a == 0,
    # i.e., d mod a == -(b+c) mod a.
    
    # Let's build: for each a, a list of (b, c, residue) where residue = (b+c) % a
    # Then for a set S, for element a in S, we check all triples (b,c,d) in S\{a}
    # by checking if there exist b,c,d in S\{a} such that (b+c) % a + d % a ≡ 0 (mod a)
    
    # Even better: for each a, precompute for each residue r in [0, a-1], the count of
    # elements in S\{a} with value ≡ r (mod a). Then we need to check if there exist
    # three distinct residues r1, r2, r3 (with multiplicities) such that r1+r2+r3 ≡ 0 (mod a)
    # and the required elements are distinct.
    
    # This is getting complex. Let me use a simpler but still fast approach:
    # For a given set S (size up to ~50), checking if it's valid takes:
    # For each a in S: O(|S|^2) pairs, then for each pair check if needed residue exists
    # Actually, for each a in S, for each pair (b,c) in S\{a}, compute r = (b+c) % a,
    # then check if there exists d in S\{a}\{b,c} with d % a == (-r) % a.
    # This is O(|S|^3) worst case but with early termination.
    
    # Let's just do the direct check but optimized with numpy or simple loops.
    # For |S| <= 50, |S|^3 = 125000, which is fast enough.
    
    def is_valid_set(S):
        """Check if set S is valid."""
        S_list = list(S)
        n = len(S_list)
        if n < 4:
            return True
        for i, a in enumerate(S_list):
            # Check if a divides sum of any 3 distinct others
            others = [S_list[j] for j in range(n) if j != i]
            m = len(others)
            if m < 3:
                continue
            # For each pair (b, c) in others, check if there's a d in others\{b,c}
            # such that (b+c+d) % a == 0
            for bi in range(m):
                b = others[bi]
                for ci in range(bi + 1, m):
                    c = others[ci]
                    rem = (b + c) % a
                    target = (-rem) % a
                    # Check if any d in others (not b, not c) has d % a == target
                    for di in range(m):
                        if di == bi or di == ci:
                            continue
                        if others[di] % a == target:
                            return False
        return True
    
    # Greedy initialization with multiple strategies
    def greedy(order):
        selected = []
        for x in order:
            new_set = selected + [x]
            if is_valid_set(new_set):
                selected.append(x)
        return selected
    
    # Try several initializations
    best = []
    
    # Descending
    s = greedy(range(N, 0, -1))
    if len(s) > len(best):
        best = s
    
    # Ascending
    s = greedy(range(1, N + 1))
    if len(s) > len(best):
        best = s
    
    # Random seeds
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        order = list(all_vals)
        random.shuffle(order)
        s = greedy(order)
        if len(s) > len(best):
            best = s
    
    # Simulated Annealing
    current = list(best)
    best_sa = list(current)
    best_size = len(current)
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    sa_start = time.time()
    sa_deadline = deadline - 5  # leave 5s for final checks
    
    while time.time() < sa_deadline and T > min_T:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap: remove one from current, add one from all_vals not in current
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            candidates = [x for x in all_vals if x not in set(current)]
            if not candidates:
                continue
            added = random.choice(candidates)
            
            new_set = current[:idx] + current[idx+1:] + [added]
            new_size = len(new_set)
            
            # Check validity
            valid = is_valid_set(new_set)
            
            if valid:
                # Accept if same size or larger
                if new_size >= len(current):
                    current = new_set
                    if len(current) > best_size:
                        best_size = len(current)
                        best_sa = list(current)
                else:
                    # Smaller: accept with probability exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        current = new_set
            # If not valid, reject
            
        elif move_type < 0.7:
            # Add: add a random element not in current
            candidates = [x for x in all_vals if x not in set(current)]
            if not candidates:
                continue
            added = random.choice(candidates)
            new_set = current + [added]
            
            if is_valid_set(new_set):
                current = new_set
                if len(current) > best_size:
                    best_size = len(current)
                    best_sa = list(current)
            # If not valid, reject (or accept with low prob? No, keep it simple)
            
        else:
            # Remove: remove a random element from current
            if len(current) <= 1:
                continue
            idx = random.randint(0, len(current) - 1)
            new_set = current[:idx] + current[idx+1:]
            # Removing always keeps validity (subset of valid set is valid)
            # But we only want to remove if it helps (allows future additions)
            # Accept with some probability to escape local optima
            if random.random() < 0.3 + 0.7 * (T / 1.0):
                current = new_set
        
        T *= cooling
    
    if len(best_sa) > len(best):
        best = best_sa
    
    # Final local search: try to add any remaining elements
    best_set = set(best)
    remaining = [x for x in all_vals if x not in best_set]
    random.shuffle(remaining)
    for x in remaining:
        if time.time() > deadline - 2:
            break
        new_set = list(best) + [x]
        if is_valid_set(new_set):
            best = new_set
            best_set.add(x)
    
    # One more pass in different order
    remaining = [x for x in all_vals if x not in set(best)]
    random.shuffle(remaining)
    for x in remaining:
        if time.time() > deadline - 1:
            break
        new_set = list(best) + [x]
        if is_valid_set(new_set):
            best = new_set
    
    return best
