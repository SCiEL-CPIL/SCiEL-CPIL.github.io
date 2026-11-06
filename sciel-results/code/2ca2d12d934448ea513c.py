import random
from math import gcd

def construct():
    # Simulated Annealing approach to maximize sum(1/m)
    # State: list of (m, r)
    
    # Precompute gcds and potential moduli
    moduli = list(range(20, 201))
    
    # Helper to check if a new (m, r) is compatible with a list of (m_i, r_i)
    def is_compatible(m, r, current_set):
        for m_i, r_i in current_set:
            g = gcd(m, m_i)
            if r % g == r_i % g:
                return False
        return True

    def get_score(current_set):
        return sum(1.0 / m for m, _ in current_set)

    # Initial state: Greedy first-fit (same as current seed)
    current_set = []
    for m in moduli:
        for r in range(m):
            if is_compatible(m, r, current_set):
                current_set.append((m, r))
                break
    
    best_set = list(current_set)
    best_score = get_score(best_set)
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    max_iters = 5000
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if T < T_min:
            break
        
        # Generate a neighbor
        # 1. Try to add a new modulus
        # 2. Try to remove a modulus
        # 3. Try to change a residue
        
        move_type = rng.random()
        new_set = list(current_set)
        
        if move_type < 0.4:
            # Add a random modulus
            m = rng.choice(moduli)
            # Check if m is already in new_set
            if any(mm == m for mm, _ in new_set):
                continue
            # Find a valid residue
            valid_rs = [r for r in range(m) if is_compatible(m, r, new_set)]
            if valid_rs:
                r = rng.choice(valid_rs)
                new_set.append((m, r))
            else:
                # Try to change an existing residue to make room? Too complex.
                # Just skip or try a different m.
                continue
        elif move_type < 0.7:
            # Remove a random modulus
            if not new_set:
                continue
            idx = rng.randrange(len(new_set))
            new_set.pop(idx)
        else:
            # Change residue of a random modulus
            if not new_set:
                continue
            idx = rng.randrange(len(new_set))
            m, r = new_set[idx]
            # Find a different valid residue
            other_rs = [r2 for r2 in range(m) if r2 != r and is_compatible(m, r2, [x for i, x in enumerate(new_set) if i != idx])]
            if other_rs:
                new_r = rng.choice(other_rs)
                new_set[idx] = (m, new_r)
            else:
                continue

        new_score = get_score(new_set)
        delta = new_score - best_score if new_score > best_score else new_score - get_score(current_set)
        
        # Acceptance probability
        # We want to maximize score.
        # If new_score > current_score, always accept.
        # Else, accept with probability exp((new_score - current_score) / T)
        
        current_score = get_score(current_set)
        if new_score >= current_score:
            current_set = new_set
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
        else:
            prob = pow(2.71828, (new_score - current_score) / T)
            if rng.random() < prob:
                current_set = new_set
        
        T *= cooling

    return [[m, r] for m, r in best_set]
