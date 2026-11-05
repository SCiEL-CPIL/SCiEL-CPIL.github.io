import math
import time
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    def score(classes):
        """Compute sum of 1/m for selected classes."""
        total = Fraction(0, 1)
        for m, r in classes:
            total += Fraction(1, m)
        return total
    
    def is_compatible(m, r, selected):
        """Check if (m, r) is compatible with all in selected."""
        for m2, r2 in selected:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    # Phase 1: Greedy construction - try all moduli, pick best residue
    # Sort by modulus ascending to prioritize small moduli (higher 1/m contribution)
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 60:
            break
        
        best_r = None
        # Try residues in order of 0, 1, ..., m-1
        for r in range(m):
            if is_compatible(m, r, selected):
                best_r = r
                break
        
        if best_r is not None:
            selected.append((m, best_r))
    
    best_score = score(selected)
    best_selected = list(selected)
    
    # Phase 2: Local search / hill climbing
    # Try swapping residues or adding/removing moduli
    time_budget = deadline - time.time() - 5  # 5 second safety margin
    
    if time_budget > 10:
        # Try to improve by reassigning residues
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            current_score = score(selected)
            
            # Randomly pick a modulus and try a different residue
            idx = random.randint(0, len(selected) - 1)
            m, r = selected[idx]
            
            # Try random residues
            found_improvement = False
            for _ in range(20):
                new_r = random.randint(0, m - 1)
                if new_r == r:
                    continue
                
                # Check compatibility with all other selected
                temp_selected = [selected[i] for i in range(len(selected)) if i != idx]
                if is_compatible(m, new_r, temp_selected):
                    # Calculate new score
                    new_selected = temp_selected + [(m, new_r)]
                    new_score = score(new_selected)
                    if new_score > current_score:
                        selected[idx] = (m, new_r)
                        current_score = new_score
                        found_improvement = True
                        break
            
            # Try adding a missing modulus
            if not found_improvement:
                missing_moduli = [m for m in range(20, 201) 
                                 if not any(m2 == m for m2, r2 in selected)]
                if missing_moduli:
                    m = random.choice(missing_moduli)
                    for _ in range(10):
                        r = random.randint(0, m - 1)
                        if is_compatible(m, r, selected):
                            selected.append((m, r))
                            new_score = score(selected)
                            if new_score > current_score:
                                current_score = new_score
                            break
            else:
                # Try removing a modulus that doesn't contribute much
                if len(selected) > 50:
                    idx = random.randint(0, len(selected) - 1)
                    m, r = selected[idx]
                    # Only remove if m is large (low contribution)
                    if m > 100:
                        new_selected = [selected[i] for i in range(len(selected)) if i != idx]
                        new_score = score(new_selected)
                        # This would decrease score, so skip
                        pass
        
        # Update best
        final_score = score(selected)
        if final_score > best_score:
            best_score = final_score
            best_selected = list(selected)
    
    # Phase 3: Simulated annealing for further improvement
    remaining_time = deadline - time.time() - 3
    if remaining_time > 20:
        sa_selected = list(best_selected)
        sa_score = best_score
        temperature = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline - 3 and temperature > 0.001:
            # Random move: change residue of a random selected modulus
            idx = random.randint(0, len(sa_selected) - 1)
            m, r = sa_selected[idx]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            # Check compatibility
            temp_selected = [sa_selected[i] for i in range(len(sa_selected)) if i != idx]
            if is_compatible(m, new_r, temp_selected):
                sa_selected[idx] = (m, new_r)
                new_score = score(sa_selected)
                delta = new_score - sa_score
                
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    sa_score = new_score
                    if sa_score > best_score:
                        best_score = sa_score
                        best_selected = list(sa_selected)
                else:
                    sa_selected[idx] = (m, old_r)
            
            temperature *= cooling_rate
    
    return [[m, r] for m, r in best_selected]
