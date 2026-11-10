import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin

    # We want to maximize sum(1/m).
    # Iterate m from 20 to 200.
    # Maintain list of selected (m, r).
    selected = []
    
    # To potentially improve, we could try multiple strategies or random restarts.
    # But simple greedy is a strong baseline.
    
    best_score = 0.0
    best_solution = []

    # Strategy 1: Greedy by increasing m
    def greedy_search(order):
        sel = []
        score = 0.0
        for m in order:
            if time.time() > deadline:
                break
            # Check gcd condition with all selected
            compatible_gcd = True
            for m_old, r_old in sel:
                if math.gcd(m, m_old) == 1:
                    compatible_gcd = False
                    break
            
            if not compatible_gcd:
                continue
            
            # Find a valid residue r
            found_r = None
            for r in range(m):
                valid = True
                for m_old, r_old in sel:
                    g = math.gcd(m, m_old)
                    if r % g == r_old % g:
                        valid = False
                        break
                if valid:
                    found_r = r
                    break
            
            if found_r is not None:
                sel.append((m, found_r))
                score += 1.0 / m
        
        return sel, score

    # Try increasing order
    sel1, score1 = greedy_search(range(20, 201))
    if score1 > best_score:
        best_score = score1
        best_solution = sel1

    # Try decreasing order? No, we want small m.
    # Try randomizing the residue choice?
    # Let's try a few random restarts with random residue selection to see if we can include more.
    import random
    random.seed(42)
    
    for _ in range(100): # 100 restarts should be fast
        if time.time() > deadline:
            break
        sel = []
        score = 0.0
        # Shuffle the order slightly? No, keep small m first.
        # Instead, randomize the residue selection.
        for m in range(20, 201):
            if time.time() > deadline:
                break
            compatible_gcd = True
            for m_old, r_old in sel:
                if math.gcd(m, m_old) == 1:
                    compatible_gcd = False
                    break
            if not compatible_gcd:
                continue
            
            # Collect all valid residues
            valid_rs = []
            for r in range(m):
                valid = True
                for m_old, r_old in sel:
                    g = math.gcd(m, m_old)
                    if r % g == r_old % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if valid_rs:
                # Pick a random valid residue
                r = random.choice(valid_rs)
                sel.append((m, r))
                score += 1.0 / m
        
        if score > best_score:
            best_score = score
            best_solution = sel

    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_solution]
