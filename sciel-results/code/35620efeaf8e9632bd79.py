import math
import time
import random

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(all_moduli)
    
    # Precompute GCD matrix for speed
    # gcds[i][j] corresponds to moduli[i] and moduli[j]
    # moduli[i] = all_moduli[i]
    # We need a fast way to check if a residue r for modulus m_i is disjoint from existing (r_k, m_k)
    # Condition: (r - r_k) % gcd(m_i, m_k) != 0
    # This is equivalent to r % gcd(m_i, m_k) != r_k % gcd(m_i, m_k)
    
    # Let's precompute for each pair (i, j) the gcd value.
    # Actually, we can just compute gcd on the fly or precompute a list of lists.
    # n_mods = 127. 127*127 = 16129. Very small.
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # Precompute "conflict constraints" for each modulus index i?
    # For a new modulus m_i, it conflicts with existing m_j if r_i % gcd(i,j) == r_j % gcd(i,j)
    
    # Heuristic for ordering:
    # Moduli with many divisors (high gcd with others) are more constrained.
    # Let's calculate a "difficulty" score for each modulus.
    # difficulty[i] = sum of gcd(all_moduli[i], all_moduli[j]) for all j != i
    # Higher difficulty -> more constraints -> place earlier?
    # Or place later?
    # Usually, in graph coloring / independent set, most constrained first (MRV) is good.
    # Here, "constrained" means it has fewer valid residues relative to others?
    # Actually, if m_i is a multiple of many others, it shares large gcds.
    # If we place a "smooth" number (highly composite) early, it restricts many others.
    # If we place a prime early, it only restricts multiples of that prime.
    # Let's try ordering by descending "sum of gcds" (most connected/constrained first).
    
    diff_scores = [0] * n_mods
    for i in range(n_mods):
        s = 0
        for j in range(n_mods):
            if i != j:
                s += gcd_matrix[i][j]
        diff_scores[i] = s
        
    # Sort indices by descending difficulty
    order_diff_desc = sorted(range(n_mods), key=lambda x: diff_scores[x], reverse=True)
    # Sort indices by ascending difficulty
    order_diff_asc = sorted(range(n_mods), key=lambda x: diff_scores[x])
    # Natural order (ascending modulus)
    order_nat = list(range(n_mods))
    # Reverse natural order (descending modulus)
    order_nat_rev = list(range(n_mods - 1, -1, -1))
    
    best_result = []
    best_len = 0

    def greedy_solve(order_indices, seed_val, randomize_residues=True):
        """
        Greedy assignment using the given order of modulus indices.
        """
        rng = random.Random(seed_val)
        assigned_residues = {} # index -> residue
        pairs = []
        
        for idx in order_indices:
            m = all_moduli[idx]
            valid_residues = []
            
            # Check all residues 0..m-1
            for r in range(m):
                ok = True
                for idx2, r2 in assigned_residues.items():
                    g = gcd_matrix[idx][idx2]
                    if (r % g) == (r2 % g):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                if randomize_residues:
                    r = rng.choice(valid_residues)
                else:
                    r = valid_residues[0] # Smallest residue
                assigned_residues[idx] = r
                pairs.append((r, m))
            else:
                # Cannot assign this modulus, skip
                pass
        
        return pairs

    # Define strategies
    # 1. Difficulty Descending (Most constrained first)
    # 2. Difficulty Ascending (Least constrained first)
    # 3. Natural Ascending
    # 4. Natural Descending
    # 5. Random shuffles
    
    strategies = [
        (order_diff_desc, 0),
        (order_diff_asc, 1),
        (order_nat, 2),
        (order_nat_rev, 3),
    ]
    
    # Add some random shuffles
    for i in range(5):
        shuffled = list(range(n_mods))
        random.Random(100 + i).shuffle(shuffled)
        strategies.append((shuffled, 100 + i))

    # Run deterministic strategies first
    for order, seed in strategies:
        if time.time() > deadline:
            break
        pairs = greedy_solve(order, seed, randomize_residues=(seed >= 100))
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_result = pairs

    # If we have time, run many random restarts with randomized residue choices
    # This helps escape local minima of the greedy choice.
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        shuffled = list(range(n_mods))
        random.Random(restart_count).shuffle(shuffled)
        
        # We can also mix in the difficulty ordering with some noise?
        # For now, pure random order with random residue selection is a strong baseline.
        # Also try difficulty-descending order with random residue selection.
        if restart_count % 2 == 0:
            order = order_diff_desc
        else:
            order = shuffled
            
        pairs = greedy_solve(order, restart_count, randomize_residues=True)
        
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_result = pairs

    # Fallback
    if not best_result:
        # Simple valid set
        best_result = [[1, 2], [0, 3], [2, 4]]

    return [[r, m] for r, m in best_result]
