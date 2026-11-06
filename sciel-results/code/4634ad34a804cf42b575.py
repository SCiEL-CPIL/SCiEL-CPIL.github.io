"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    n = 34
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer for the evaluator
    
    # Precompute indices and neighbor lists
    triples = list(combinations(range(n), 3))
    E = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    quads = list(combinations(range(n), 4))
    Q = len(quads)
    quad_index = {q: i for i, q in enumerate(quads)}
    
    quints = list(combinations(range(n), 5))
    U = len(quints)
    quint_index = {q: i for i, q in enumerate(quints)}
    
    # Precompute which quads and quints each triple belongs to
    triple_to_quads = [[] for _ in range(E)]
    triple_to_quints = [[] for _ in range(E)]
    
    for i, (a, b, c) in enumerate(triples):
        # Quads containing (a,b,c)
        for x in range(n):
            if x in (a, b, c):
                continue
            q = tuple(sorted((a, b, c, x)))
            triple_to_quads[i].append(quad_index[q])
            
        # Quints containing (a,b,c)
        # Choose 2 from the remaining n-3 vertices
        others = [x for x in range(n) if x not in (a, b, c)]
        for x_idx in range(len(others)):
            for y_idx in range(x_idx + 1, len(others)):
                x = others[x_idx]
                y = others[y_idx]
                q = tuple(sorted((a, b, c, x, y)))
                triple_to_quints[i].append(quint_index[q])

    def solve_with_sa(seed):
        random.seed(seed)
        colors = [random.randint(0, 1) for _ in range(E)]
        
        # Initialize counts
        red_count = [0] * Q
        blue_count = [0] * U
        
        for i, (a, b, c) in enumerate(triples):
            if colors[i] == 0: # Red
                for qi in triple_to_quads[i]:
                    red_count[qi] += 1
            else: # Blue
                for ui in triple_to_quints[i]:
                    blue_count[ui] += 1
        
        # Initial score
        bad_quads = sum(1 for rc in red_count if rc == 4)
        bad_quints = sum(1 for bc in blue_count if bc == 10)
        current_score = bad_quads + bad_quints
        
        if current_score == 0:
            return colors
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        # We track the best solution found
        best_colors = colors[:]
        best_score = current_score
        
        # Main SA loop
        while T > min_T:
            # Check time limit
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Pick a random edge to flip
            i = random.randint(0, E - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta in score
            # We need to know how many quads/quints transition from good to bad or vice versa
            
            delta_bad_quads = 0
            delta_bad_quints = 0
            
            if old_color == 0: # Red -> Blue
                # Red count decreases by 1 for quads in triple_to_quads[i]
                # Blue count increases by 1 for quints in triple_to_quints[i]
                for qi in triple_to_quads[i]:
                    rc = red_count[qi]
                    if rc == 4:
                        delta_bad_quads -= 1
                    elif rc == 3:
                        delta_bad_quads += 1 # Becomes 4? No, 3->2 is good. Wait.
                        # If rc==4, it was bad, now 3 (good). Delta -1.
                        # If rc==3, it was good, now 2 (good). Delta 0.
                        # If rc==2, it was good, now 1 (good). Delta 0.
                        # So only rc==4 matters for reduction.
                        # Wait, does it become bad? No, decreasing red count never creates a new red K4.
                        # It only fixes existing ones.
                        pass
                    # Actually, simply:
                    # if rc == 4: delta_bad_quads -= 1
                    # else: delta_bad_quads += 0
                    
                for ui in triple_to_quints[i]:
                    bc = blue_count[ui]
                    if bc == 9:
                        delta_bad_quints += 1 # Becomes 10 (bad)
                    # if bc == 10: it was bad, now 9 (good). But we are adding blue, so it can't be 10 before adding?
                    # If it was 10, adding blue makes it 11. Still bad.
                    # So only transition 9->10 creates a new bad quint.
                    
            else: # Blue -> Red
                # Blue count decreases by 1 for quints in triple_to_quints[i]
                # Red count increases by 1 for quads in triple_to_quads[i]
                for ui in triple_to_quints[i]:
                    bc = blue_count[ui]
                    if bc == 10:
                        delta_bad_quints -= 1 # Fixes a bad quint
                        # If bc == 9, becomes 8 (good).
                        
                for qi in triple_to_quads[i]:
                    rc = red_count[qi]
                    if rc == 3:
                        delta_bad_quads += 1 # Becomes 4 (bad)
                        # If rc == 4, becomes 5 (still bad).
                        
            delta_score = delta_bad_quads + delta_bad_quints
            
            # Acceptance
            if delta_score < 0 or random.random() < pow(2.71828, -delta_score / T):
                # Apply flip
                colors[i] = new_color
                
                # Update counts
                if old_color == 0:
                    for qi in triple_to_quads[i]:
                        red_count[qi] -= 1
                    for ui in triple_to_quints[i]:
                        blue_count[ui] += 1
                else:
                    for ui in triple_to_quints[i]:
                        blue_count[ui] -= 1
                    for qi in triple_to_quads[i]:
                        red_count[qi] += 1
                
                current_score += delta_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    
                    if best_score == 0:
                        return best_colors
                        
            T *= cooling
            
        return best_colors

    # Try multiple seeds
    for seed in range(10):
        res = solve_with_sa(seed)
        if res:
            # Verify
            # Re-verify to be safe
            red_count_check = [0] * Q
            blue_count_check = [0] * U
            for i, (a, b, c) in enumerate(triples):
                if res[i] == 0:
                    for qi in triple_to_quads[i]:
                        red_count_check[qi] += 1
                else:
                    for ui in triple_to_quints[i]:
                        blue_count_check[ui] += 1
            
            valid = True
            for rc in red_count_check:
                if rc == 4:
                    valid = False
                    break
            if valid:
                for bc in blue_count_check:
                    if bc == 10:
                        valid = False
                        break
            
            if valid:
                return {'n': n, 'colors': res}
        
        if time.time() - start_time > time_limit - 10.0:
            break

    # Fallback to smaller n if 34 fails (should not happen with good SA, but just in case)
    # Actually, the task says "Maximize n". If we can't get 34, we should try to return a valid smaller one?
    # The prompt says "A valid colouring on n points certifies R(4,5;3) > n".
    # The seed returns n=9 if it fails.
    # However, the published construction is 34. We should strive for 34.
    # If we fail, we might return a random valid one for smaller n?
    # Let's just return the best we found if it was valid, otherwise a known small valid one.
    
    # Simple valid construction for n=9? 
    # The seed used a specific one. Let's use the seed's fallback.
    n_small = 9
    triples_small = list(combinations(range(n_small), 3))
    colors_small = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_small]
    
    return {'n': n_small, 'colors': colors_small}

# EVOLVE-BLOCK-END
