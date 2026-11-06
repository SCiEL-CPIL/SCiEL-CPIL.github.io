import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Precompute masks for all triples
    all_triple_masks = [sum(1 << p for p in t) for t in all_triples]
    
    def get_alpha_fast(triples):
        """
        Compute the independence number efficiently using bitmask operations.
        """
        if not triples:
            return 16
            
        t_masks = [sum(1 << p for p in t) for t in triples]
        n = len(t_masks)
        
        # Check sizes from 16 down to 1
        for k in range(16, 0, -1):
            for comb in itertools.combinations(range(16), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                # Check if this set contains any triple
                independent = True
                for tm in t_masks:
                    if (mask & tm) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 0

    # Strategy:
    # 1. Try a deterministic construction based on affine geometry
    # 2. Local search with simulated annealing

    best_alpha = 16
    best_system = []

    # Deterministic construction: Use a structured approach
    # Try to construct a system with small independence number
    def try_deterministic():
        """
        Try a structured construction based on dividing points into groups
        and creating triples that cover all possible independent sets.
        """
        # One approach: use a Steiner system-like construction
        # For 16 points, we can try to create triples that hit all large subsets
        
        # Simple structured approach: create triples from different "layers"
        triples = []
        
        # Layer 1: Triples from first 8 points
        for i in range(0, 8, 3):
            if i + 2 < 8:
                triples.append(tuple(sorted([i, i+1, i+2])))
        
        # Layer 2: Triples mixing first 8 and last 8
        for i in range(4):
            triples.append(tuple(sorted([i, 8+i, 8+(i+1)%8])))
        
        # Layer 3: Triples from last 8 points
        for i in range(8, 16, 3):
            if i + 2 < 16:
                triples.append(tuple(sorted([i, i+1, i+2])))
        
        # Add more triples to reduce independence number
        # Cross-layer triples
        for i in range(8):
            for j in range(8):
                if (i, j) not in [(t[0], t[1]) for t in triples]:
                    candidate = (i, 8+j, (i+j)%8)
                    # Check if valid (no pair conflicts)
                    valid = True
                    for t in triples:
                        common = len(set(candidate) & set(t))
                        if common > 1:
                            valid = False
                            break
                    if valid:
                        triples.append(candidate)
        
        # Remove duplicates and sort
        triples = list(set(triples))
        
        # Verify linearity (any two triples share at most one point)
        valid_triples = []
        for t in triples:
            valid = True
            for t2 in valid_triples:
                if len(set(t) & set(t2)) > 1:
                    valid = False
                    break
            if valid:
                valid_triples.append(t)
        
        return valid_triples

    # Try deterministic first
    det_system = try_deterministic()
    if det_system:
        det_alpha = get_alpha_fast(det_system)
        if det_alpha < best_alpha:
            best_alpha = det_alpha
            best_system = det_system[:]

    # Initialize with random greedy packings
    for _ in range(30):
        if time.time() - start_time > deadline:
            break
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        current = []
        used_pairs = set()
        for t in shuffled:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)

        alpha = get_alpha_fast(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]

        if best_alpha <= 3:
            break

    current_system = best_system[:]
    current_alpha = best_alpha

    # Local search with simulated annealing
    def can_add_fast(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    iteration = 0
    no_improve_count = 0
    while time.time() - start_time < deadline:
        iteration += 1
        no_improve_count += 1

        # Temperature decay for simulated annealing
        accept_prob = max(0.01, 1.0 / (1.0 + iteration / 1000.0))

        move_type = random.choice(['add', 'swap', 'add', 'swap'])  # Bias toward add/swap

        if move_type == 'add':
            # Find valid triples to add
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])

            # Find candidates: triples that don't use any used pair
            candidates = [t for t in all_triples if all(p not in used_pairs for p in triple_pairs[t])]

            if candidates:
                # Try multiple candidates and pick the best
                best_new_alpha = current_alpha
                best_new_system = current_system
                for _ in range(min(10, len(candidates))):
                    new_t = random.choice(candidates)
                    new_system = current_system + [new_t]
                    new_alpha = get_alpha_fast(new_system)
                    if new_alpha < best_new_alpha:
                        best_new_alpha = new_alpha
                        best_new_system = new_system
                
                if best_new_alpha < current_alpha:
                    current_system = best_new_system
                    current_alpha = best_new_alpha
                    no_improve_count = 0
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif best_new_alpha == current_alpha:
                    if random.random() < accept_prob:
                        current_system = best_new_system

        elif move_type == 'swap':
            if not current_system:
                continue

            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]

            # Build used_pairs excluding t_remove
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])

            # Candidates: triples that don't use any used pair (and are not t_remove)
            candidates = [t for t in all_triples if all(p not in used_pairs for p in triple_pairs[t])]

            if candidates:
                # Try multiple candidates and pick the best
                best_new_alpha = current_alpha
                best_new_system = current_system
                for _ in range(min(10, len(candidates))):
                    new_t = random.choice(candidates)
                    if new_t == t_remove:
                        continue
                    new_system = current_system[:]
                    new_system[idx] = new_t
                    new_alpha = get_alpha_fast(new_system)
                    if new_alpha < best_new_alpha:
                        best_new_alpha = new_alpha
                        best_new_system = new_system
                
                if best_new_alpha < current_alpha:
                    current_system = best_new_system
                    current_alpha = best_new_alpha
                    no_improve_count = 0
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif best_new_alpha == current_alpha:
                    if random.random() < accept_prob:
                        current_system = best_new_system

        # Occasional random restart to escape deep local minima
        if iteration % 1000 == 0 and best_alpha > 3:
            # Try a fresh greedy initialization
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            restart_sys = []
            used_pairs = set()
            for t in shuffled:
                if all(p not in used_pairs for p in triple_pairs[t]):
                    restart_sys.append(t)
                    for p in triple_pairs[t]:
                        used_pairs.add(p)
            r_alpha = get_alpha_fast(restart_sys)
            if r_alpha < current_alpha:
                current_system = restart_sys
                current_alpha = r_alpha
                no_improve_count = 0
                if r_alpha < best_alpha:
                    best_alpha = r_alpha
                    best_system = restart_sys[:]

        if best_alpha <= 3:
            break

    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
