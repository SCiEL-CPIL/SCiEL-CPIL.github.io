"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill

    def is_sidon(elems):
        n = len(elems)
        seen = set()
        for i in range(n):
            for j in range(i, n):
                s = elems[i] + elems[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def max_sidon_size(S):
        n = len(S)
        for size in range(n, 0, -1):
            for subset in combinations(range(n), size):
                elems = [S[i] for i in subset]
                if is_sidon(elems):
                    return size
        return 1

    def count_collisions(S):
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                sums[s] = sums.get(s, 0) + 1
        total = 0
        for cnt in sums.values():
            if cnt > 1:
                total += cnt * (cnt - 1) // 2
        return total

    def collision_density(S):
        """Higher density means more forced collisions per pair."""
        n = len(S)
        total_pairs = n * (n + 1) // 2
        coll = count_collisions(S)
        return coll / total_pairs if total_pairs > 0 else 0

    def max_collision_sum_count(S):
        """Find the sum value with the most representations."""
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                sums[s] = sums.get(s, 0) + 1
        return max(sums.values()) if sums else 0

    rng = random.Random(42)

    # Strategy: Build sets with deliberate additive structure to force collisions.
    # Key insight: use clustered values in a small range to create many sum collisions.
    # Then use local search to minimize max Sidon subset while maximizing collision density.

    # Generate candidate base structures with high collision potential
    def generate_base_structures():
        structures = []
        
        # Structure 1: Arithmetic progression (max collisions)
        for start in range(0, 117):
            for step in [1, 2, 3]:
                if start + 11 * step <= 128:
                    structures.append(list(range(start, start + 12 * step, step)))
        
        # Structure 2: Two clusters (e.g., two small APs)
        for c1_start in range(0, 60):
            for c2_start in range(60, 117):
                if c1_start + 5 <= 128 and c2_start + 5 <= 128:
                    s = list(range(c1_start, c1_start + 6)) + list(range(c2_start, c2_start + 6))
                    if len(set(s)) == 12:
                        structures.append(s)
        
        # Structure 3: Clustered around a few values
        for center in range(5, 124):
            offsets = list(range(-5, 6))
            s = [center + o for o in offsets if 0 <= center + o <= 128]
            if len(s) >= 12:
                s = s[:12]
                if len(set(s)) == 12:
                    structures.append(s)
        
        # Structure 4: Random small-range sets
        for _ in range(50):
            s = [rng.randrange(0, 26) for _ in range(12)]
            while len(set(s)) < 12:
                s = [rng.randrange(0, 26) for _ in range(12)]
            structures.append(s)
        
        # Structure 5: Random medium-range sets
        for _ in range(50):
            s = [rng.randrange(0, 40) for _ in range(12)]
            while len(set(s)) < 12:
                s = [rng.randrange(0, 40) for _ in range(12)]
            structures.append(s)
        
        return structures

    # Initialize with best from base structures
    best_set = list(range(12))
    best_ms = max_sidon_size(best_set)
    best_coll = count_collisions(best_set)

    base_structs = generate_base_structures()
    for s in base_structs:
        ms = max_sidon_size(s)
        coll = count_collisions(s)
        if ms < best_ms or (ms == best_ms and coll > best_coll):
            best_ms = ms
            best_set = list(s)
            best_coll = coll

    # Local search: swap-based to escape local minima
    current = list(best_set)
    current_ms = best_ms
    current_coll = best_coll

    no_improve_count = 0
    max_no_improve = 5000

    # Define move operators
    def swap_move(s, i, j, new_val):
        """Replace element at position i with new_val, ensuring distinctness."""
        trial = list(s)
        old_val = trial[i]
        trial[i] = new_val
        # Check distinctness
        if len(set(trial)) < 12:
            return None
        return trial

    def neighborhood(s, range_max):
        """Generate neighboring states by changing one element."""
        moves = []
        for i in range(12):
            for _ in range(5):  # Sample 5 random values per position
                new_val = rng.randrange(range_max)
                if new_val != s[i]:
                    trial = swap_move(s, i, None, new_val)
                    if trial is not None:
                        moves.append(trial)
        return moves

    # Main search loop
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 3:
            break

        inner_steps = min(5000, int(remaining * 150))
        if inner_steps < 10:
            break

        for _ in range(inner_steps):
            if time.time() > deadline - 2:
                break
            
            # Choose a random position to modify
            i = rng.randrange(12)
            old_val = current[i]
            
            # Choose new value from a range that encourages collisions
            # Use smaller ranges more often
            r = rng.random()
            if r < 0.5:
                range_max = 26  # [0, 25]
            elif r < 0.8:
                range_max = 40  # [0, 39]
            else:
                range_max = 60  # [0, 59]
            
            new_val = rng.randrange(range_max)
            if new_val == old_val or new_val in current:
                continue

            trial = list(current)
            trial[i] = new_val
            
            # Evaluate: prefer lower max_sidon_size, then higher collision count
            trial_ms = max_sidon_size(trial)
            trial_coll = count_collisions(trial)
            
            # Accept if better or equal with higher collisions
            if trial_ms < current_ms or (trial_ms == current_ms and trial_coll > current_coll):
                current = trial
                current_ms = trial_ms
                current_coll = trial_coll
                no_improve_count = 0
                
                if current_ms < best_ms or (current_ms == best_ms and current_coll > best_coll):
                    best_ms = current_ms
                    best_set = list(current)
                    best_coll = current_coll
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart from a perturbed version of best or a new base structure
            r = rng.random()
            if r < 0.4:
                # Perturb best set
                current = list(best_set)
                for _ in range(rng.randint(2, 5)):
                    i = rng.randrange(12)
                    current[i] = rng.randrange(40)
                    while current[i] in current[:i] or current[i] in current[i+1:]:
                        current[i] = rng.randrange(40)
            elif r < 0.7:
                # Pick a random base structure
                current = list(rng.choice(base_structs))
            else:
                # Random restart in small range
                current = [rng.randrange(26) for _ in range(12)]
                while len(set(current)) < 12:
                    current = [rng.randrange(26) for _ in range(12)]
            
            current_ms = max_sidon_size(current)
            current_coll = count_collisions(current)
            no_improve_count = 0

    # Final high-precision polish: exhaustive local search around best_set
    if time.time() < deadline - 5:
        polish_steps = 3000
        for _ in range(polish_steps):
            if time.time() > deadline - 2:
                break
            i = rng.randrange(12)
            old_val = best_set[i]
            new_val = rng.randrange(30)  # Tighter range for polish
            if new_val == old_val or new_val in best_set:
                continue
            trial = list(best_set)
            trial[i] = new_val
            ms = max_sidon_size(trial)
            coll = count_collisions(trial)
            if ms < best_ms or (ms == best_ms and coll > best_coll):
                best_ms = ms
                best_set = trial
                best_coll = coll

    return best_set
# EVOLVE-BLOCK-END
