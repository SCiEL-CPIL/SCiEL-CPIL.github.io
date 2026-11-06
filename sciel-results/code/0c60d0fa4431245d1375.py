"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    # weights[i] = 1/denominators[i] as Fraction
    weights = [Fraction(1, d) for d in denominators]
    one = Fraction(1, 1)
    
    best_indices = None
    best_deficit = None
    
    def current_deficit(indices):
        s = sum(weights[i] for i in indices)
        return one - s  # positive since sum < 1
    
    def valid(indices):
        s = sum(weights[i] for i in indices)
        return 0 < s < one
    
    def try_improve(indices, time_limit):
        """Local search with add, remove, 1-1, 1-2, 2-1 moves."""
        nonlocal best_indices, best_deficit
        indices = set(indices)
        if not valid(indices):
            return indices
        deficit = current_deficit(indices)
        if best_deficit is None or deficit < best_deficit:
            best_deficit = deficit
            best_indices = list(indices)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_move = None
            best_move_deficit = deficit
            
            # Try add moves
            for i in range(n):
                if i not in indices:
                    new_set = indices | {i}
                    s = sum(weights[j] for j in new_set)
                    if s < one:
                        d = one - s
                        if d < best_move_deficit:
                            best_move_deficit = d
                            best_move = ('add', i, None)
            
            # Try remove moves
            for i in list(indices):
                new_set = indices - {i}
                if new_set:  # must be nonempty
                    s = sum(weights[j] for j in new_set)
                    if s < one:
                        d = one - s
                        if d < best_move_deficit:
                            best_move_deficit = d
                            best_move = ('remove', i, None)
            
            # Try 1-1 swap moves
            for i in list(indices):
                for j in range(n):
                    if j not in indices:
                        new_set = (indices - {i}) | {j}
                        s = sum(weights[k] for k in new_set)
                        if s < one:
                            d = one - s
                            if d < best_move_deficit:
                                best_move_deficit = d
                                best_move = ('swap', i, j)
            
            # Try 1-2 moves (remove 1, add 2)
            for i in list(indices):
                for j in range(n):
                    if j not in indices:
                        for k in range(n):
                            if k not in indices and k != j:
                                new_set = (indices - {i}) | {j, k}
                                s = sum(weights[m] for m in new_set)
                                if s < one:
                                    d = one - s
                                    if d < best_move_deficit:
                                        best_move_deficit = d
                                        best_move = ('swap12', i, (j, k))
            
            # Try 2-1 moves (remove 2, add 1)
            for i in list(indices):
                for i2 in list(indices):
                    if i2 != i:
                        for j in range(n):
                            if j not in indices:
                                new_set = (indices - {i, i2}) | {j}
                                if new_set:
                                    s = sum(weights[m] for m in new_set)
                                    if s < one:
                                        d = one - s
                                        if d < best_move_deficit:
                                            best_move_deficit = d
                                            best_move = ('swap21', (i, i2), j)
            
            if best_move is not None:
                improved = True
                mtype = best_move[0]
                if mtype == 'add':
                    indices.add(best_move[1])
                elif mtype == 'remove':
                    indices.discard(best_move[1])
                elif mtype == 'swap':
                    indices.discard(best_move[1])
                    indices.add(best_move[2])
                elif mtype == 'swap12':
                    indices.discard(best_move[1])
                    indices.add(best_move[2][0])
                    indices.add(best_move[2][1])
                elif mtype == 'swap21':
                    indices.discard(best_move[1][0])
                    indices.discard(best_move[1][1])
                    indices.add(best_move[2])
                deficit = best_move_deficit
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(indices)
        
        return indices
    
    def greedy_init():
        """Greedy: add smallest fractions first to get close to 1."""
        indices = set()
        total = Fraction(0)
        # Sort by weight descending (largest reciprocals first)
        for i in sorted(range(n), key=lambda x: -weights[x]):
            if total + weights[i] < one:
                indices.add(i)
                total += weights[i]
        return indices
    
    def random_init():
        """Random subset with sum < 1."""
        for _ in range(100):
            indices = set()
            total = Fraction(0)
            for i in random.sample(range(n), n):
                if total + weights[i] < one:
                    indices.add(i)
                    total += weights[i]
            if indices:
                return indices
        return greedy_init()
    
    # Initial greedy
    init_set = greedy_init()
    if valid(init_set):
        try_improve(init_set, deadline)
    
    # Multiple randomized restarts
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        # Diverse initialization
        if restart_count % 3 == 1:
            init_set = random_init()
        elif restart_count % 3 == 2:
            # Complement-ish: start with large denominators
            indices = set()
            total = Fraction(0)
            for i in sorted(range(n), key=lambda x: weights[x]):  # smallest weights first
                if total + weights[i] < one:
                    indices.add(i)
                    total += weights[i]
            init_set = indices
        else:
            # Random subset
            k = rng.randint(5, 30)
            init_set = set(rng.sample(range(n), k))
            # Ensure validity
            total = sum(weights[i] for i in init_set)
            if total >= one:
                # Remove until valid
                for i in sorted(init_set, key=lambda x: -weights[x]):
                    if total >= one:
                        total -= weights[i]
                        init_set.discard(i)
            if not init_set:
                init_set = greedy_init()
        
        if valid(init_set):
            try_improve(init_set, deadline)
        
        # Periodic check
        if time.time() > deadline - 2:
            break
    
    if best_indices is None or not best_indices:
        # Fallback: single largest denominator
        best_indices = [0]  # denominator 2, sum = 1/2, deficit = 1/2
    
    return best_indices
# EVOLVE-BLOCK-END
