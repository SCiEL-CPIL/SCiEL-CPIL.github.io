"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Efficient polish using coverage counts
    def build_coverage(w, n):
        counts = defaultdict(int)
        for i in range(len(w) - n + 1):
            counts[tuple(w[i:i+n])] += 1
        return counts
    
    def try_remove(improved, counts, pos, n):
        """Try to remove position pos from improved. Returns True if removed."""
        j_min = max(0, pos - n + 1)
        j_max = min(pos, len(improved) - n)
        
        safe = True
        affected = []
        for j in range(j_min, j_max + 1):
            w = tuple(improved[j:j+n])
            affected.append(w)
            if counts[w] <= 1:
                safe = False
                break
        
        if safe:
            for w in affected:
                counts[w] -= 1
            improved.pop(pos)
            return True
        return False
    
    def greedy_remove_pass(improved, counts, n, order="fwd"):
        """Do one greedy removal pass in the given order. Returns count of removals."""
        removals = 0
        if order == "fwd":
            i = 0
            while i < len(improved):
                if try_remove(improved, counts, i, n):
                    removals += 1
                else:
                    i += 1
        elif order == "bwd":
            i = len(improved) - 1
            while i >= 0:
                if try_remove(improved, counts, i, n):
                    removals += 1
                else:
                    i -= 1
        elif order == "random":
            # Collect all positions and try in random order, rebuilding as we go
            # This is more expensive but can find different removals
            positions = list(range(len(improved)))
            random.shuffle(positions)
            for pos in positions:
                if 0 <= pos < len(improved):
                    if try_remove(improved, counts, pos, n):
                        removals += 1
        return removals
    
    improved = list(word)
    counts = build_coverage(improved, n)
    
    # Phase 1: Standard greedy removal passes (fwd, bwd, random)
    while time.time() < deadline:
        changed = False
        for order in ["fwd", "bwd", "random"]:
            if time.time() > deadline:
                break
            r = greedy_remove_pass(improved, counts, n, order)
            if r > 0:
                changed = True
        if not changed:
            break
    
    if time.time() > deadline:
        return "".join(map(str, improved))
    
    # Phase 2: Simulated Annealing for Deletion
    # We maintain `counts` of permutations.
    # A move is to remove a random character.
    # If the resulting word is valid (all counts > 0), we always accept.
    # If invalid, we accept with probability exp(-delta_cost / T).
    # Cost = length + penalty * missing_count.
    
    # Pre-compute all permutations for fast missing check?
    # No, we just track `missing_count`.
    # `missing_count` is the number of permutations with count 0.
    
    # Calculate initial missing count
    # Since we started with a valid word, missing_count should be 0.
    # But let's verify.
    target_perms = set(itertools.permutations(range(1, n+1)))
    # Actually, we don't need the set, just the count of keys in `counts` with value > 0.
    # But `counts` only contains keys that appear.
    # So missing = total_perms - len(counts).
    total_perms = 1
    for k in range(2, n+1):
        total_perms *= k
    
    def get_missing(counts):
        return total_perms - len(counts)
    
    current_missing = get_missing(counts)
    current_len = len(improved)
    current_cost = current_len + current_missing * 10000.0 # High penalty for missing perms
    
    best_len = current_len
    best_word = improved[:]
    best_counts = dict(counts)
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        if T < 0.1:
            break
            
        # Pick a random position to try removing
        if len(improved) <= n:
            break
            
        pos = random.randint(0, len(improved) - 1)
        
        # Try removing pos
        # Affected windows: [pos-n+1, pos]
        j_min = max(0, pos - n + 1)
        j_max = min(pos, len(improved) - n)
        
        affected = []
        safe = True
        for j in range(j_min, j_max + 1):
            w = tuple(improved[j:j+n])
            affected.append(w)
            if counts[w] <= 1:
                safe = False
        
        if safe:
            # Apply removal
            for w in affected:
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            removed_char = improved.pop(pos)
            
            new_len = len(improved)
            new_missing = get_missing(counts)
            new_cost = new_len + new_missing * 10000.0
            
            # Accept if better or with probability
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                # Accept
                current_len = new_len
                current_missing = new_missing
                current_cost = new_cost
                
                if new_missing == 0 and new_len < best_len:
                    best_len = new_len
                    best_word = improved[:]
                    best_counts = dict(counts)
            else:
                # Revert
                improved.insert(pos, removed_char)
                for w in affected:
                    counts[w] = counts.get(w, 0) + 1
        else:
            # Cannot remove safely (would make a perm missing)
            # We could still accept if we allow missing, but for now, skip
            # To allow SA to escape local minima, we might want to allow it.
            # Let's allow it with penalty.
            # But we need to update counts correctly even if it becomes 0.
            # The `safe` check above prevents us from deleting keys that go to 0?
            # No, `safe` is just a flag.
            # Let's force the removal and update counts, allowing 0s.
            
            # Re-do the update allowing 0s
            for w in affected:
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            removed_char = improved.pop(pos)
            
            new_len = len(improved)
            new_missing = get_missing(counts)
            new_cost = new_len + new_missing * 10000.0
            
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                current_len = new_len
                current_missing = new_missing
                current_cost = new_cost
                
                if new_missing == 0 and new_len < best_len:
                    best_len = new_len
                    best_word = improved[:]
                    best_counts = dict(counts)
            else:
                improved.insert(pos, removed_char)
                for w in affected:
                    counts[w] = counts.get(w, 0) + 1
        
        T *= cooling
        
        # Periodic greedy cleanup
        if random.random() < 0.01:
            # Run a fast greedy pass
            # This might break the SA state, so we should rebuild counts?
            # Or just run it and update counts.
            # greedy_remove_pass updates `counts` in place.
            # But it modifies `improved`.
            # This is a "move" in SA terms, but a complex one.
            # Let's just run it and update best if valid.
            old_len = len(improved)
            greedy_remove_pass(improved, counts, n, "random")
            if len(improved) < old_len:
                current_len = len(improved)
                current_missing = get_missing(counts)
                current_cost = current_len + current_missing * 10000.0
                if current_missing == 0 and current_len < best_len:
                    best_len = current_len
                    best_word = improved[:]
                    best_counts = dict(counts)

    # Final greedy removal pass on the best word found
    improved = best_word[:]
    counts = best_counts
    
    for order in ["fwd", "bwd", "random"]:
        if time.time() > deadline - 1:
            break
        greedy_remove_pass(improved, counts, n, order)
    
    # Phase 3: Local Search with Swaps and Insertions
    # Try to improve further by moving characters around
    if time.time() < deadline - 2:
        # Try swapping adjacent characters
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            if len(improved) < 2:
                break
            i = random.randint(0, len(improved) - 2)
            # Try swapping i and i+1
            improved[i], improved[i+1] = improved[i+1], improved[i]
            
            # Rebuild counts (expensive but simple for now)
            new_counts = build_coverage(improved, n)
            new_missing = get_missing(new_counts)
            
            if new_missing == 0:
                counts = new_counts
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_word = improved[:]
                    best_counts = dict(counts)
            else:
                # Revert
                improved[i], improved[i+1] = improved[i+1], improved[i]
        
        # Try moving a character to a different position
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            if len(improved) < 3:
                break
            i = random.randint(0, len(improved) - 1)
            j = random.randint(0, len(improved) - 1)
            if i == j:
                continue
            
            # Move char at i to position j
            char = improved.pop(i)
            if j > i:
                j -= 1
            improved.insert(j, char)
            
            # Rebuild counts
            new_counts = build_coverage(improved, n)
            new_missing = get_missing(new_counts)
            
            if new_missing == 0:
                counts = new_counts
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_word = improved[:]
                    best_counts = dict(counts)
            else:
                # Revert
                improved.pop(j)
                improved.insert(i, char)
    
    # Final validation and return
    # Ensure we have a valid word
    final_counts = build_coverage(improved, n)
    if get_missing(final_counts) > 0:
        # Fallback to best_word if improved is invalid
        improved = best_word[:]
    
    return "".join(map(str, improved))
# EVOLVE-BLOCK-END
