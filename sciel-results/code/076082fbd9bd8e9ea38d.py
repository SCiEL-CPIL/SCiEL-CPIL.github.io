import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, then a+a is a square, so a cannot be in the set.
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict adjacency using bitmasks
    # adj[i] is a bitmask of all j != i such that valid_numbers[i] + valid_numbers[j] is a square
    adj = [0] * num_valid
    for i, a in enumerate(valid_numbers):
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def verify(result):
        """Verify that the result is a valid independent set."""
        result_set = set(result)
        for a in result:
            if 2 * a in squares:
                return False
            for b in result:
                if a <= b and (a + b) in squares:
                    return False
        return True

    def greedy_min_degree():
        """Standard min-degree greedy for maximum independent set."""
        remaining = (1 << num_valid) - 1
        selected = 0
        while remaining:
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (adj[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                    if deg == 0:
                        break
                v ^= lowbit
            if best_v != -1:
                selected |= (1 << best_v)
                remaining &= ~(adj[best_v] | (1 << best_v))
            else:
                break
        return selected

    def greedy_max_cardinality():
        """Greedy that picks vertices with maximum potential (min conflicts)."""
        remaining = (1 << num_valid) - 1
        selected = 0
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (adj[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                    if deg == 0:
                        break
                v ^= lowbit
            if best_v != -1:
                selected |= (1 << best_v)
                remaining &= ~(adj[best_v] | (1 << best_v))
            else:
                break
        return selected

    def local_search_1swap(initial_selected, iterations=2000):
        """Try to improve by removing one element and adding another."""
        selected_mask = initial_selected
        # Create a set for fast lookup
        selected_set = set()
        v = selected_mask
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            selected_set.add(idx)
            v ^= lowbit

        for _ in range(iterations):
            improved = False
            # Try removing each selected element
            for rem_idx in list(selected_set):
                # What can we add back if we remove rem_idx?
                # Candidates are non-selected elements that are only blocked by rem_idx
                # i.e., candidates not in selected_set, and their only conflict in selected_set is rem_idx
                candidates = []
                for add_idx in range(num_valid):
                    if add_idx in selected_set:
                        continue
                    # Check if add_idx conflicts with selected_set
                    conflicts = adj[add_idx] & selected_mask
                    # If the only conflict is rem_idx, then removing rem_idx allows adding add_idx
                    if conflicts == (1 << rem_idx):
                        candidates.append(add_idx)
                
                if not candidates:
                    continue
                
                # Pick the best candidate (one with fewest conflicts overall)
                best_add = max(candidates, key=lambda x: -(adj[x] & ~(selected_mask | (1 << rem_idx))).bit_count())
                
                # Actually, we want to minimize new conflicts after adding
                # But since we're just doing 1-swap, the size stays the same
                # This doesn't help. We need to think differently.
                
            # Alternative: Try to add a vertex by removing multiple
            # This is more complex. Let's try a simpler approach:
            # Randomly try to remove a random selected vertex and see if we can add more than one.
            
            if not improved and time.time() > deadline - 5:
                break
        
        return selected_mask

    def random_greedy(seed):
        """Greedy with randomized tie-breaking."""
        rng = random.Random(seed)
        remaining = (1 << num_valid) - 1
        selected = 0
        while remaining:
            # Find all vertices with minimum degree
            min_deg = float('inf')
            candidates = []
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (adj[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                    if deg == 0:
                        break
                elif deg == min_deg:
                    candidates.append(idx)
                v ^= lowbit
            
            if candidates:
                # Pick randomly among candidates
                best_v = rng.choice(candidates)
                selected |= (1 << best_v)
                remaining &= ~(adj[best_v] | (1 << best_v))
            else:
                break
        return selected

    def two_phase_greedy():
        """Two-phase greedy: first phase picks low-degree, second phase picks from remaining."""
        # Phase 1: Standard min-degree
        phase1 = greedy_min_degree()
        
        # Phase 2: Try to add more from remaining using a different strategy
        remaining = ((1 << num_valid) - 1) & ~phase1
        # Remove neighbors of selected
        for i in range(num_valid):
            if (phase1 >> i) & 1:
                remaining &= ~adj[i]
        
        # Now try to greedily add from remaining
        phase2 = phase1
        while remaining:
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (adj[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                    if deg == 0:
                        break
                v ^= lowbit
            if best_v != -1:
                phase2 |= (1 << best_v)
                remaining &= ~(adj[best_v] | (1 << best_v))
            else:
                break
        return phase2

    # Run multiple strategies and keep the best
    best_mask = 0
    best_size = 0

    # Strategy 1: Standard min-degree
    mask = greedy_min_degree()
    if mask.bit_count() > best_size:
        best_size = mask.bit_count()
        best_mask = mask

    # Strategy 2: Two-phase greedy
    mask = two_phase_greedy()
    if mask.bit_count() > best_size:
        best_size = mask.bit_count()
        best_mask = mask

    # Strategy 3: Random restarts with different seeds
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        mask = random_greedy(seed)
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask

    # Strategy 4: Greedy with different orderings (e.g., by value)
    # Try picking in order of valid_numbers
    for order in [list(range(num_valid)), list(range(num_valid-1, -1, -1))]:
        if time.time() > deadline - 5:
            break
        selected = 0
        for idx in order:
            # Check if idx can be added (no conflicts with selected)
            if not (adj[idx] & selected):
                selected |= (1 << idx)
        if selected.bit_count() > best_size:
            best_size = selected.bit_count()
            best_mask = selected

    # Convert best_mask to result
    result = []
    v = best_mask
    while v:
        lowbit = v & (-v)
        idx = lowbit.bit_length() - 1
        result.append(valid_numbers[idx])
        v ^= lowbit

    if not result:
        return [1]

    # Verify
    if verify(result):
        return sorted(result)
    else:
        # Fallback
        fallback = []
        for a in valid_numbers:
            can_add = True
            for b in fallback:
                if (a + b) in squares or (2 * a) in squares:
                    can_add = False
                    break
            if can_add:
                fallback.append(a)
        if fallback:
            return sorted(fallback)
        return [1]

if __name__ == "__main__":
    print(construct())
