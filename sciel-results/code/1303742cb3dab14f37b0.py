import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build adjacency bitsets (conflict graph)
    adj = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                adj[i] |= (1 << j)

    all_mask = (1 << num_valid) - 1

    # Greedy min-degree initialization
    def greedy_min_degree():
        remaining = all_mask
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
                v ^= lowbit
            selected |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return selected

    # Branch and bound MIS solver with bit-parallelism
    def mis_bnb(initial_mask, time_limit, rng):
        """Branch and bound for maximum independent set."""
        best = [0]  # best size found
        best_set = [0]  # best set as bitset
        
        def expand(current_set, candidates, depth):
            if time.time() > time_limit:
                return
            
            # Upper bound: current_size + number of candidates
            current_size = current_set.bit_count()
            if current_size + candidates.bit_count() <= best[0]:
                return
            
            if candidates == 0:
                if current_size > best[0]:
                    best[0] = current_size
                    best_set[0] = current_set
                return
            
            # Greedy coloring upper bound on candidates
            # Simple bound: use maximum degree heuristic
            # Pick a vertex with max degree in candidates
            v = candidates
            max_deg = 0
            pivot = -1
            temp = candidates
            while temp:
                lowbit = temp & (-temp)
                idx = lowbit.bit_length() - 1
                deg = (adj[idx] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = idx
                temp ^= lowbit
            
            if pivot == -1:
                if current_size > best[0]:
                    best[0] = current_size
                    best_set[0] = current_set
                return
            
            # Branch on pivot: include or exclude
            # Include pivot
            new_candidates = candidates & ~(adj[pivot] | (1 << pivot))
            new_set = current_set | (1 << pivot)
            expand(new_set, new_candidates, depth + 1)
            
            # Exclude pivot
            new_candidates2 = candidates & ~(1 << pivot)
            expand(current_set, new_candidates2, depth + 1)
        
        # Initial greedy solution
        greedy_set = 0
        remaining = initial_mask
        while remaining:
            # Pick vertex with min degree
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
                v ^= lowbit
            greedy_set |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        
        best[0] = greedy_set.bit_count()
        best_set[0] = greedy_set
        
        # Start B&B from full candidate set
        expand(0, initial_mask, 0)
        
        return best_set[0]

    # Multiple restarts with different random orderings
    rng = random.Random(42)
    best_overall = 0
    best_overall_set = 0
    
    # First, get a good initial solution
    initial = greedy_min_degree()
    best_overall = initial.bit_count()
    best_overall_set = initial
    
    # Try branch and bound with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Run B&B with a portion of the time
        bnb_time = min(remaining_time * 0.7, 40.0)
        result = mis_bnb(all_mask, start_time + bnb_time, rng)
        if result.bit_count() > best_overall:
            best_overall = result.bit_count()
            best_overall_set = result
    
    # Final local search: try to add any vertex that fits
    for _ in range(5000):
        improved = False
        for i in range(num_valid):
            if not (best_overall_set >> i) & 1:
                if not (adj[i] & best_overall_set):
                    best_overall_set |= (1 << i)
                    best_overall += 1
                    improved = True
        if not improved:
            break
    
    # Convert to list of values
    result = []
    v = best_overall_set
    while v:
        lowbit = v & (-v)
        idx = lowbit.bit_length() - 1
        result.append(valid_numbers[idx])
        v ^= lowbit

    result.sort()

    # Verify validity
    result_set = set(result)
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
        for b in result:
            if a <= b and (a + b) in squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: rebuild from bitset carefully
        result = []
        v = best_overall_set
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            val = valid_numbers[idx]
            if 2 * val not in squares:
                # Check all existing
                ok = True
                for existing in result:
                    if val + existing in squares:
                        ok = False
                        break
                if ok:
                    result.append(val)
            v ^= lowbit
        result.sort()
    
    return result
