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

    rng = random.Random(42)

    best_set = list(range(12))
    best_ms = max_sidon_size(best_set)

    current = list(best_set)
    current_coll = count_collisions(current)

    iterations = 0
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        improved = False
        # Dynamic inner loop size based on remaining time
        remaining = deadline - time.time()
        inner_steps = min(1000, int(remaining * 50))
        if inner_steps < 10:
            break

        for _ in range(inner_steps):
            if time.time() > deadline - 2:
                break
            i = rng.randrange(12)
            old_val = current[i]
            new_val = rng.randrange(129)
            if new_val == old_val or new_val in current:
                continue

            current[i] = new_val
            coll = count_collisions(current)
            
            # Only compute expensive max_sidon_size if collision count is promising
            # or periodically to ensure we don't miss a better ms with similar coll
            if coll > current_coll:
                # Check if this is a potential new best
                # Heuristic: if coll is significantly higher, it's likely better
                # But we must verify ms
                ms = max_sidon_size(current)
                if ms < best_ms:
                    best_ms = ms
                    best_set = list(current)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                current_coll = coll
                improved = True
            else:
                current[i] = old_val
                no_improve_count += 1

        if not improved or no_improve_count > max_no_improve:
            # Restart strategy: perturb best set or random restart
            if rng.random() < 0.5:
                # Perturb best set
                current = list(best_set)
                for _ in range(rng.randint(2, 5)):
                    i = rng.randrange(12)
                    current[i] = rng.randrange(129)
                    while current[i] in current[:i] or current[i] in current[i+1:]:
                        current[i] = rng.randrange(129)
            else:
                # Random restart
                current = [rng.randrange(129) for _ in range(12)]
                # Ensure distinct
                while len(set(current)) < 12:
                    current = [rng.randrange(129) for _ in range(12)]
            
            current_coll = count_collisions(current)
            no_improve_count = 0

    return best_set
# EVOLVE-BLOCK-END
