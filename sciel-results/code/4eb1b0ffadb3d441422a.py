import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moduli in [20, 200]
    moduli_range = range(20, 201)
    all_moduli = list(moduli_range)
    
    # Precompute gcds
    # gcd_cache[m1][m2]
    # Using a dict for sparse storage or just computing on the fly?
    # 181 moduli. 181*181/2 ~ 16000 pairs.
    # Computing on the fly is fast enough with math.gcd.
    
    def is_valid_assignment(selected):
        """
        selected: dict {m: r}
        Returns True if all pairs are disjoint.
        """
        keys = list(selected.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                m1 = keys[i]
                m2 = keys[j]
                r1 = selected[m1]
                r2 = selected[m2]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True

    def get_conflicts(selected):
        """
        Returns a list of conflicting pairs (m1, m2).
        """
        conflicts = []
        keys = list(selected.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                m1 = keys[i]
                m2 = keys[j]
                r1 = selected[m1]
                r2 = selected[m2]
                g = math.gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    conflicts.append((m1, m2))
        return conflicts

    def solve_residues(moduli_set):
        """
        Given a set of moduli, find valid residues.
        Uses local search.
        """
        if not moduli_set:
            return {}
        
        # Initialize randomly
        selected = {m: random.randint(0, m-1) for m in moduli_set}
        
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            conflicts = get_conflicts(selected)
            if not conflicts:
                return selected
            
            # Pick a random conflict
            m1, m2 = random.choice(conflicts)
            
            # Try to fix m1
            if random.random() < 0.5:
                m_to_fix = m1
            else:
                m_to_fix = m2
            
            # Find a valid residue for m_to_fix
            # Valid means: for all other m, r_new not equiv r_other mod gcd
            found = False
            for _ in range(100): # Try 100 random residues
                r_new = random.randint(0, m_to_fix - 1)
                valid = True
                for m_other, r_other in selected.items():
                    if m_other == m_to_fix:
                        continue
                    g = math.gcd(m_to_fix, m_other)
                    if (r_new - r_other) % g == 0:
                        valid = False
                        break
                if valid:
                    selected[m_to_fix] = r_new
                    found = True
                    break
            
            if not found:
                # If we can't find a valid residue easily, maybe remove m_to_fix?
                # Or just try another conflict.
                pass
        
        # If we didn't solve it, return empty or best effort?
        # For scoring, we only count valid sets.
        if is_valid_assignment(selected):
            return selected
        else:
            return {}

    def evaluate(moduli_set):
        """
        Returns score for a set of moduli if a valid residue assignment exists.
        """
        selected = solve_residues(moduli_set)
        if not selected:
            return 0.0, {}
        
        score = sum(1.0 / m for m in moduli_set)
        return score, selected

    best_score = 0.0
    best_classes = []
    
    # Candidate 1: All even moduli
    even_moduli = [m for m in all_moduli if m % 2 == 0]
    score, classes = evaluate(even_moduli)
    if score > best_score:
        best_score = score
        best_classes = [[m, r] for m, r in classes.items()]
        
    # Candidate 2: All multiples of 3
    mod3_moduli = [m for m in all_moduli if m % 3 == 0]
    score, classes = evaluate(mod3_moduli)
    if score > best_score:
        best_score = score
        best_classes = [[m, r] for m, r in classes.items()]
        
    # Candidate 3: All multiples of 5
    mod5_moduli = [m for m in all_moduli if m % 5 == 0]
    score, classes = evaluate(mod5_moduli)
    if score > best_score:
        best_score = score
        best_classes = [[m, r] for m, r in classes.items()]
        
    # Candidate 4: Local Search on Moduli Set
    # Start with even_moduli and try to swap out/in moduli to improve score.
    current_set = set(even_moduli)
    current_score, current_classes = evaluate(current_set)
    
    if current_score > best_score:
        best_score = current_score
        best_classes = [[m, r] for m, r in current_classes.items()]
        
    # Simulated Annealing on the set of moduli
    T = 0.05
    for step in range(5000):
        if time.time() > deadline:
            break
            
        # Pick a random modulus to toggle
        m_rand = random.choice(all_moduli)
        
        if m_rand in current_set:
            # Remove
            new_set = current_set - {m_rand}
        else:
            # Add
            new_set = current_set | {m_rand}
            
        # Evaluate new set
        new_score, new_classes = evaluate(new_set)
        
        # Delta score
        delta = new_score - current_score
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = new_set
            current_score = new_score
            current_classes = new_classes
            
            if current_score > best_score:
                best_score = current_score
                best_classes = [[m, r] for m, r in current_classes.items()]
        
        T *= 0.999

    return best_classes
