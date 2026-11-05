import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    min_m, max_m = 20, 200
    all_moduli = list(range(min_m, max_m + 1))
    inv_m = {m: 1.0 / m for m in all_moduli}

    def is_compatible(m, r, classes):
        for m_i, r_i in classes:
            g = math.gcd(m, m_i)
            if g > 1 and r % g == r_i % g:
                return False
        return True

    def find_valid_residue(m, classes, randomize=False):
        # Collect valid residues
        valids = []
        for r in range(m):
            if is_compatible(m, r, classes):
                valids.append(r)
                if not randomize:
                    return r
        if valids:
            return random.choice(valids)
        return None

    def get_score(classes):
        return sum(inv_m[m] for m, _ in classes)

    def random_greedy():
        classes = []
        # Randomize the order of moduli to get diverse starts
        mods = all_moduli[:]
        random.shuffle(mods)
        # But we want small moduli to have priority?
        # If we shuffle, we might pick 200 first, then 20.
        # 200 has many residues, so it's easy.
        # 20 has few.
        # To maximize score, we should prioritize small moduli.
        # So let's sort by m, but with some randomness in tie-breaking?
        # Or just sort by m.
        # Let's try sorting by m, but randomizing the residue choice.
        mods_sorted = all_moduli[:]
        for m in mods_sorted:
            r = find_valid_residue(m, classes, randomize=True)
            if r is not None:
                classes.append([m, r])
        return classes

    best_classes = random_greedy()
    best_score = get_score(best_classes)
    
    # Run SA
    current_classes = [c[:] for c in best_classes]
    current_score = best_score
    T = 1.0
    cooling = 0.999
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if T < 1e-5:
            T = 1e-5
        
        if no_improve_count > max_no_improve:
            # Restart
            new_start = random_greedy()
            new_score = get_score(new_start)
            if new_score > best_score:
                best_score = new_score
                best_classes = [c[:] for c in new_start]
                current_classes = [c[:] for c in new_start]
                current_score = new_score
            else:
                # Keep current if it's better than the new random start?
                # Or just restart current from best?
                current_classes = [c[:] for c in best_classes]
                current_score = best_score
            
            T = 1.0
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.5) # Increase patience
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Add
            current_mods = {m for m, _ in current_classes}
            missing = [m for m in all_moduli if m not in current_mods]
            if missing:
                m_new = random.choice(missing)
                r_new = find_valid_residue(m_new, current_classes, randomize=True)
                if r_new is not None:
                    current_classes.append([m_new, r_new])
                    delta = inv_m[m_new]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
        elif move_type < 0.7:
            # Remove
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                m_rem = current_classes[idx][0]
                delta = -inv_m[m_rem]
                # Accept if better or with prob
                if delta > 0 or random.random() < math.exp(-delta / T):
                    current_classes.pop(idx)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        else:
            # Perturb
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                m_cur = current_classes[idx][0]
                r_cur = current_classes[idx][1]
                
                others = current_classes[:idx] + current_classes[idx+1:]
                valids = []
                for r in range(m_cur):
                    if r != r_cur and is_compatible(m_cur, r, others):
                        valids.append(r)
                
                if valids:
                    new_r = random.choice(valids)
                    current_classes[idx][1] = new_r
                    # Score unchanged, but structure changed
                    # This is an exploratory move
                    no_improve_count += 1 # Count as non-improvement for restart logic

        T *= cooling

    return best_classes
