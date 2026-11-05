import time
from fractions import Fraction
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    recips_float = [1.0 / d for d in denominators]
    recips_frac = [Fraction(1, d) for d in denominators]

    best_deficit = None
    best_indices = None
    best_mask = 0
    best_sum_f = 0.0

    def mask_to_indices(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_deficit_frac_from_mask(mask):
        s = Fraction(0)
        for i in range(n):
            if (mask >> i) & 1:
                s += recips_frac[i]
        if s >= 1:
            return None
        return 1 - s

    def get_sum_float_from_mask(mask):
        s = 0.0
        for i in range(n):
            if (mask >> i) & 1:
                s += recips_float[i]
        return s

    def update_best(mask, sum_f):
        nonlocal best_deficit, best_indices, best_mask, best_sum_f
        d = get_deficit_frac_from_mask(mask)
        if d is not None:
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = mask_to_indices(mask)
                best_mask = mask
                best_sum_f = sum_f

    # 1. Greedy Initialization
    greedy_mask = 0
    total_f = 0.0
    for i in range(n):
        if total_f + recips_float[i] < 1.0:
            greedy_mask |= (1 << i)
            total_f += recips_float[i]
    update_best(greedy_mask, total_f)

    # 2. Known Good Solution: 1/2 + 1/3 + 1/7 + 1/43
    # Indices: 0, 1, 5, 41
    known_mask = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 41)
    update_best(known_mask, get_sum_float_from_mask(known_mask))

    # 3. Exhaustive search for small sizes (3-5) to find good local optima
    for size in range(3, 6):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            s_f = 0.0
            for i in combo:
                mask |= (1 << i)
                s_f += recips_float[i]
            if s_f < 1.0:
                update_best(mask, s_f)

    # 4. Random Initializations and Repair
    for _ in range(200):
        if time.time() > deadline:
            break
        k = random.randint(5, 25)
        subset = random.sample(range(n), k)
        mask = 0
        s_f = 0.0
        for i in subset:
            mask |= (1 << i)
            s_f += recips_float[i]
        
        # Repair: remove random items until sum < 1
        while s_f >= 1.0 and mask != 0:
            bits = [i for i in range(n) if (mask >> i) & 1]
            if not bits:
                break
            rem = random.choice(bits)
            mask &= ~(1 << rem)
            s_f -= recips_float[rem]
            
        if mask != 0:
            update_best(mask, s_f)

    if best_indices is None:
        return [0]

    # 5. Simulated Annealing
    T = 1e-3
    T_min = 1e-9
    cooling = 0.9995
    
    current_mask = best_mask
    current_sum_f = best_sum_f
    current_deficit = 1.0 - best_sum_f

    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add one element
            available = [i for i in range(n) if not ((current_mask >> i) & 1)]
            if not available:
                continue
            i_add = random.choice(available)
            new_mask = current_mask | (1 << i_add)
            new_sum_f = current_sum_f + recips_float[i_add]
            if new_sum_f < 1.0:
                new_deficit = 1.0 - new_sum_f
                if new_deficit < current_deficit:
                    current_mask = new_mask
                    current_sum_f = new_sum_f
                    current_deficit = new_deficit
                    update_best(new_mask, new_sum_f)
                else:
                    # Accept worse with probability
                    delta = new_deficit - current_deficit
                    if random.random() < pow(2.71828, -delta / T):
                        current_mask = new_mask
                        current_sum_f = new_sum_f
                        current_deficit = new_deficit
            # Else: invalid, reject
            
        elif move_type < 0.6:
            # Remove one element
            current_bits = [i for i in range(n) if (current_mask >> i) & 1]
            if not current_bits:
                continue
            i_rem = random.choice(current_bits)
            new_mask = current_mask & ~(1 << i_rem)
            new_sum_f = current_sum_f - recips_float[i_rem]
            # Removing always keeps it valid (if it was valid)
            new_deficit = 1.0 - new_sum_f
            if new_deficit < current_deficit:
                current_mask = new_mask
                current_sum_f = new_sum_f
                current_deficit = new_deficit
                update_best(new_mask, new_sum_f)
            else:
                delta = new_deficit - current_deficit
                if random.random() < pow(2.71828, -delta / T):
                    current_mask = new_mask
                    current_sum_f = new_sum_f
                    current_deficit = new_deficit
                    
        else:
            # Swap one element
            current_bits = [i for i in range(n) if (current_mask >> i) & 1]
            available = [i for i in range(n) if not ((current_mask >> i) & 1)]
            if not current_bits or not available:
                continue
            i_out = random.choice(current_bits)
            i_in = random.choice(available)
            new_mask = (current_mask & ~(1 << i_out)) | (1 << i_in)
            new_sum_f = current_sum_f - recips_float[i_out] + recips_float[i_in]
            if new_sum_f < 1.0:
                new_deficit = 1.0 - new_sum_f
                if new_deficit < current_deficit:
                    current_mask = new_mask
                    current_sum_f = new_sum_f
                    current_deficit = new_deficit
                    update_best(new_mask, new_sum_f)
                else:
                    delta = new_deficit - current_deficit
                    if random.random() < pow(2.71828, -delta / T):
                        current_mask = new_mask
                        current_sum_f = new_sum_f
                        current_deficit = new_deficit

        # Cool down
        T *= cooling
        if T < T_min:
            # Restart from best
            current_mask = best_mask
            current_sum_f = best_sum_f
            current_deficit = 1.0 - best_sum_f
            T = 1e-3

    return best_indices
