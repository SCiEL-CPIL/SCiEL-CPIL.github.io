"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Precompute 1/m as Fraction for exact arithmetic
    inv = {m: Fraction(1, m) for m in candidates}
    
    # Greedy initialization: pick moduli in order of smallest m (highest 1/m),
    # with residue 0 for each, checking pairwise disjointness.
    # Two classes [m1, r1] and [m2, r2] are disjoint iff r1 != r2 mod gcd(m1, m2).
    # With r=0 for all, they're disjoint iff 0 != 0 mod gcd(m1,m2), i.e., gcd(m1,m2) != 1.
    # Wait, that's wrong. r1=0, r2=0: 0 != 0 mod g is false for any g >= 1.
    # So all residues 0 are NOT disjoint (they overlap at 0 mod g for any g).
    # We need r_i != r_j mod gcd(m_i, m_j) for all pairs.
    
    # Better greedy: process moduli in order of 1/m (descending), pick residue that
    # is disjoint from all already-picked classes.
    
    def is_disjoint(m_new, r_new, existing):
        for (m_e, r_e) in existing:
            g = gcd(m_new, m_e)
            if r_new % g == r_e % g:
                return False
        return True
    
    def greedy_init():
        selected = []
        # Sort by m ascending (higher 1/m first)
        for m in sorted(candidates):
            if time.time() > deadline:
                break
            # Try residues 0..m-1, pick first valid one
            found = False
            for r in range(m):
                if is_disjoint(m, r, selected):
                    selected.append((m, r))
                    found = True
                    break
            if not found:
                continue
        return selected
    
    best = greedy_init()
    best_score = sum(inv[m] for m, r in best)
    
    # Simulated annealing to improve
    # State: set of (m, r) pairs
    # Move: remove one, add one (with new residue), or change residue of one
    
    def compute_score(state):
        return sum(inv[m] for m, r in state)
    
    def try_add(state, m, r):
        """Try to add (m, r) to state, return new state if valid, else None."""
        if m in [x[0] for x in state]:
            return None  # moduli must be distinct
        for (m_e, r_e) in state:
            g = gcd(m, m_e)
            if r % g == r_e % g:
                return None
        return state + [(m, r)]
    
    def try_remove(state, idx):
        return state[:idx] + state[idx+1:]
    
    # SA parameters
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    random.seed(42)
    
    while T > T_min and time.time() < deadline:
        state = best
        score = best_score
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Random move
            move_type = random.choice(['add', 'remove', 'change_r'])
            
            if move_type == 'add':
                # Pick a modulus not in state
                used_mods = set(m for m, r in state)
                available = [m for m in candidates if m not in used_mods]
                if not available:
                    continue
                m_new = random.choice(available)
                # Try random residues
                for attempt in range(10):
                    r_new = random.randint(0, m_new - 1)
                    new_state = try_add(state, m_new, r_new)
                    if new_state:
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < pow(2.718281828, delta / T):
                            state, score = new_state, new_score
                        break
            elif move_type == 'remove':
                if len(state) <= 1:
                    continue
                idx = random.randint(0, len(state) - 1)
                new_state = try_remove(state, idx)
                new_score = compute_score(new_state)
                delta = new_score - score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    state, score = new_state, new_score
            elif move_type == 'change_r':
                if not state:
                    continue
                idx = random.randint(0, len(state) - 1)
                m_old, r_old = state[idx]
                # Try a new residue for this modulus
                for attempt in range(10):
                    r_new = random.randint(0, m_old - 1)
                    if r_new == r_old:
                        continue
                    # Check disjointness with all other elements
                    valid = True
                    for j, (m_e, r_e) in enumerate(state):
                        if j == idx:
                            continue
                        g = gcd(m_old, m_e)
                        if r_new % g == r_e % g:
                            valid = False
                            break
                    if valid:
                        new_state = state[:idx] + [(m_old, r_new)] + state[idx+1:]
                        new_score = compute_score(new_state)
                        delta = new_score - score
                        if delta > 0 or random.random() < pow(2.718281828, delta / T):
                            state, score = new_state, new_score
                        break
        
        if score > best_score:
            best, best_score = state, score
        
        T *= cooling
    
    # Convert to required format
    result = [[m, r] for m, r in best]
    return result
# EVOLVE-BLOCK-END
