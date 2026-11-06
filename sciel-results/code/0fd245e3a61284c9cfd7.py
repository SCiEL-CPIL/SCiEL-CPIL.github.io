"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def diff_set(s):
        """Return set of all positive differences in sorted set s."""
        n = len(s)
        return {s[j] - s[i] for i in range(n) for j in range(i + 1, n)}
    
    def is_sidon(s):
        """Check if set has all distinct positive differences."""
        n = len(s)
        d = [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]
        return len(set(d)) == len(d)
    
    def covers_1_to_30(s):
        """Check if all differences 1..30 are present."""
        d = diff_set(s)
        return all(i in d for i in range(1, 31))
    
    def greedy_construct(seed=0):
        """Greedy construction with randomization."""
        rng = random.Random(seed)
        a = [0]
        for d in range(1, 31):
            if time.time() > deadline:
                break
            if d in diff_set(a):
                continue
            found = False
            # Try adding x = v +/- d for existing v, with some randomization
            candidates = list(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            rng.shuffle(candidates)
            for x in candidates:
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                # Fallback: add two elements y, y+d in a high range
                y = 2 * max(a) + 3 * d + 1
                attempts = 0
                while time.time() < deadline and attempts < 1000:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        break
                    y += 1
                    attempts += 1
        return a
    
    def try_remove_element(a):
        """Try to remove an element if the set still covers 1..30 and is Sidon."""
        for i in range(len(a) - 1, 0, -1):  # Don't remove 0
            trial = a[:i] + a[i+1:]
            if covers_1_to_30(trial) and is_sidon(trial):
                return trial
        return a
    
    def local_search(a, max_iterations=5000):
        """Local search to reduce the largest element."""
        a = sorted(a)
        best_a = list(a)
        best_max = max(a)
        no_improve = 0
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            a = sorted(best_a)
            
            # Strategy 0: Try to remove an unnecessary element
            removed = try_remove_element(a)
            if len(removed) < len(a):
                best_a = removed
                best_max = max(best_a)
                improved = True
            
            # Strategy 1: Try to replace the largest element with a smaller one
            if not improved and len(a) > 1:
                largest = a[-1]
                # Try a range of smaller values
                for new_val in range(largest - 1, max(0, largest - 500), -1):
                    if new_val in a[:-1]:
                        continue
                    trial = sorted(a[:-1] + [new_val])
                    if is_sidon(trial) and covers_1_to_30(trial):
                        best_a = trial
                        best_max = max(trial)
                        improved = True
                        break
            
            # Strategy 2: Try shifting individual elements down
            if not improved:
                a = sorted(best_a)
                for i in range(len(a) - 1, 0, -1):
                    if a[i] > 1:
                        new_val = a[i] - 1
                        if new_val not in a:
                            trial = list(a)
                            trial[i] = new_val
                            trial = sorted(trial)
                            if is_sidon(trial) and covers_1_to_30(trial):
                                best_a = trial
                                best_max = max(trial)
                                improved = True
                                break
            
            # Strategy 3: Try moving an element to a much smaller position
            if not improved:
                a = sorted(best_a)
                for i in range(len(a) - 1, 0, -1):
                    # Try to move a[i] to a random smaller position
                    for _ in range(10):
                        new_val = random.randint(1, a[i] - 1) if a[i] > 1 else 0
                        if new_val in a[:i] or new_val in a[i+1:]:
                            continue
                        trial = sorted(a[:i] + [new_val] + a[i+1:])
                        if is_sidon(trial) and covers_1_to_30(trial):
                            if max(trial) < best_max:
                                best_a = trial
                                best_max = max(trial)
                                improved = True
                                break
                    if improved:
                        break
            
            # Strategy 4: Try swapping two elements
            if not improved:
                a = sorted(best_a)
                for i in range(1, len(a) - 1):
                    for j in range(i + 1, len(a)):
                        trial = list(a)
                        trial[i], trial[j] = trial[j], trial[i]
                        trial = sorted(trial)
                        if is_sidon(trial) and covers_1_to_30(trial):
                            if max(trial) < best_max:
                                best_a = trial
                                best_max = max(trial)
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve >= 50:
                    break
            else:
                no_improve = 0
        
        return best_a
    
    # Step 1: Try multiple greedy constructions with different seeds
    best_a = None
    best_max = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        a = greedy_construct(seed)
        if a and covers_1_to_30(a) and is_sidon(a):
            if max(a) < best_max:
                best_max = max(a)
                best_a = a
    
    if best_a is None:
        # Fallback: use first greedy result
        best_a = greedy_construct(0)
    
    # Step 2: Local search to minimize largest element
    if time.time() < deadline - 15:
        best_a = local_search(best_a, max_iterations=8000)
    
    # Step 3: More aggressive local search if time remains
    if time.time() < deadline - 5:
        best_a = local_search(best_a, max_iterations=15000)
    
    # Final validation
    if not covers_1_to_30(best_a) or not is_sidon(best_a):
        # Should not happen, but fallback
        best_a = [0]
        for d in range(1, 31):
            if d in diff_set(best_a):
                continue
            found = False
            for x in range(1, 10000):
                trial = sorted(best_a + [x])
                if is_sidon(trial):
                    best_a = trial
                    found = True
                    break
            if not found:
                # Add two elements
                y = 2 * max(best_a) + 3 * d + 1
                while time.time() < deadline:
                    trial = sorted(best_a + [y, y + d])
                    if is_sidon(trial):
                        best_a = trial
                        break
                    y += 1
    
    return best_a
# EVOLVE-BLOCK-END
