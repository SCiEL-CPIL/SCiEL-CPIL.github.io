import time
import random
import math
from collections import defaultdict

# Precomputation
N = 256

def get_ap_masks(term_count):
    masks = []
    max_d = (N - 1) // (term_count - 1)
    for d in range(1, max_d + 1):
        max_a = N - (term_count - 1) * d
        for a in range(1, max_a + 1):
            mask = 0
            for j in range(term_count):
                val = a + j * d
                mask |= (1 << (val - 1))
            masks.append(mask)
    return masks

M3 = get_ap_masks(3)
M4 = get_ap_masks(4)

# Precompute 3-APs and 4-APs containing each element x (0-indexed)
ap3_by_elem = [[] for _ in range(N)]
ap4_by_elem = [[] for _ in range(N)]

for i, m in enumerate(M3):
    for x in range(N):
        if (m >> x) & 1:
            ap3_by_elem[x].append(i)

for i, m in enumerate(M4):
    for x in range(N):
        if (m >> x) & 1:
            ap4_by_elem[x].append(i)

# Precompute degree of 4-AP conflicts for each element
elem_4ap_degree = [len(ap4_by_elem[i]) for i in range(N)]

# For faster 3-AP counting and delta calculation:
# For each element x, store list of (other1, other2, ap_idx) where {x, o1, o2} is a 3-AP
# We only count if x is the smallest to avoid double counting.
ap3_partners = [[] for _ in range(N)]
for i, m in enumerate(M3):
    bits = [b for b in range(N) if (m >> b) & 1]
    # bits are sorted? get_ap_masks iterates a, d. val = a + j*d.
    # So bits are [a, a+d, a+2d]. Sorted.
    for pos in range(3):
        x = bits[pos]
        others = [bits[j] for j in range(3) if j != pos]
        # Only store if x is the smallest? 
        # In count, we check x < o1 and x < o2.
        # It's better to pre-filter: only store partners where x is the smallest.
        if x < others[0] and x < others[1]:
             ap3_partners[x].append((others[0], others[1]))

# For 4-AP validity checking
ap4_other_mask = [[] for _ in range(N)]
for i, m in enumerate(M4):
    bits = [b for b in range(N) if (m >> b) & 1]
    for pos in range(4):
        x = bits[pos]
        others_mask = 0
        for j in range(4):
            if j != pos:
                others_mask |= (1 << bits[j])
        ap4_other_mask[x].append(others_mask)

def count_3aps(mask):
    count = 0
    for x in range(N):
        if (mask >> x) & 1:
            for (o1, o2) in ap3_partners[x]:
                if (mask >> o1) & 1 and (mask >> o2) & 1:
                    count += 1
    return count

def has_4ap(mask):
    for x in range(N):
        if (mask >> x) & 1:
            for others_mask in ap4_other_mask[x]:
                if (mask & others_mask) == others_mask:
                    return True
    return False

def is_valid_addition(mask, x):
    bit = 1 << x
    if mask & bit:
        return False
    for others_mask in ap4_other_mask[x]:
        if (mask & others_mask) == others_mask:
            return False
    return True

def calc_delta_add(mask, x):
    delta = 0
    for (o1, o2) in ap3_partners[x]:
        if (mask >> o1) & 1 and (mask >> o2) & 1:
            delta += 1
    return delta

def calc_delta_remove(mask, x):
    delta = 0
    for (o1, o2) in ap3_partners[x]:
        if (mask >> o1) & 1 and (mask >> o2) & 1:
            delta += 1
    return delta

def greedy_init(seed=None):
    if seed is not None:
        random.seed(seed)
    
    mask = 0
    order = list(range(N))
    random.shuffle(order)
    
    for x in order:
        if is_valid_addition(mask, x):
            mask |= (1 << x)
            
    return mask

def local_search_sa(mask, max_time=10.0, seed=None):
    """Simulated Annealing / Randomized Local Search"""
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    best_mask = mask
    best_count = count_3aps(mask)
    current_mask = mask
    current_count = best_count
    
    temp = 10.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        # Pick a random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Find valid additions
            candidates = []
            for x in range(N):
                if not (current_mask >> x) & 1:
                    if is_valid_addition(current_mask, x):
                        candidates.append(x)
            if not candidates:
                continue
            x = random.choice(candidates)
            new_mask = current_mask | (1 << x)
            delta = calc_delta_add(current_mask, x)
            new_count = current_count + delta
            accept = True
            if new_count < current_count:
                # Accept worse moves with probability exp(-delta_T / temp)
                delta_T = current_count - new_count
                if random.random() < math.exp(-delta_T / temp):
                    accept = True
                else:
                    accept = False
            if accept:
                current_mask = new_mask
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask
        elif move_type == 'remove':
            candidates = [x for x in range(N) if (current_mask >> x) & 1]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_mask = current_mask & ~(1 << x)
            delta = calc_delta_remove(current_mask, x)
            new_count = current_count - delta
            accept = True
            if new_count < current_count:
                delta_T = current_count - new_count
                if random.random() < math.exp(-delta_T / temp):
                    accept = True
                else:
                    accept = False
            if accept:
                current_mask = new_mask
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask
        else: # swap
            in_set = [x for x in range(N) if (current_mask >> x) & 1]
            out_set = [x for x in range(N) if not (current_mask >> x) & 1]
            if not in_set or not out_set:
                continue
            x_in = random.choice(in_set)
            x_out = random.choice(out_set)
            
            # Check if x_out can be added if x_in is removed
            # This is complex. Simpler: just try add x_out, remove x_in.
            # Validity of new set:
            # 1. x_out must not form 4-AP with (current_mask without x_in)
            # 2. Removing x_in doesn't break 4-APs (it only helps)
            
            temp_mask = current_mask & ~(1 << x_in)
            if is_valid_addition(temp_mask, x_out):
                new_mask = temp_mask | (1 << x_out)
                # Calculate delta
                # Remove x_in: lose 3-APs involving x_in
                d_rem = calc_delta_remove(current_mask, x_in)
                # Add x_out: gain 3-APs involving x_out in the new context
                # Note: 3-APs involving both x_in and x_out are tricky.
                # But x_in is removed. So we only count 3-APs in new_mask.
                # Simpler: just recount? No, too slow.
                # Approximation:
                d_add = calc_delta_add(temp_mask, x_out)
                # This ignores 3-APs that had x_in and x_out.
                # Given sparsity, this is a good heuristic.
                new_count = current_count - d_rem + d_add
                
                accept = True
                if new_count < current_count:
                    delta_T = current_count - new_count
                    if random.random() < math.exp(-delta_T / temp):
                        accept = True
                    else:
                        accept = False
                if accept:
                    current_mask = new_mask
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_mask = current_mask
                            
        temp *= cooling
        if temp < 0.1:
            temp = 1.0 # Restart temperature
            
    return best_mask, best_count

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    best_mask = 0
    best_count = 0
    
    # Phase 1: Greedy Initializations
    num_greeds = 50
    for i in range(num_greeds):
        if time.time() > deadline - 20:
            break
        m = greedy_init(seed=i)
        c = count_3aps(m)
        if c > best_count:
            best_count = c
            best_mask = m
            
    if best_mask == 0:
        # Fallback: simple valid set
        best_mask = (1 << 0) | (1 << 1) | (1 << 2)
        best_count = 1
        
    # Phase 2: Local Search / SA from best
    if time.time() < deadline - 5:
        best_mask, best_count = local_search_sa(best_mask, max_time=min(20, deadline - time.time() - 5), seed=42)
        
    # Phase 3: Restarts with Perturbation
    remaining = deadline - time.time()
    if remaining > 5:
        num_restarts = max(1, int(remaining / 10))
        for i in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            # Perturb best_mask
            m = best_mask
            # Remove 10-20% of elements
            elements = [x for x in range(N) if (m >> x) & 1]
            random.shuffle(elements)
            num_rem = max(1, len(elements) // 5)
            for x in elements[:num_rem]:
                m &= ~(1 << x)
                
            # Add random valid elements
            for _ in range(num_rem * 2):
                x = random.randint(0, N-1)
                if is_valid_addition(m, x):
                    m |= (1 << x)
            
            # Refine
            m, c = local_search_sa(m, max_time=min(5, deadline - time.time() - 1), seed=100+i)
            if c > best_count:
                best_count = c
                best_mask = m
                
    # Ensure non-empty
    if best_mask == 0:
        best_mask = (1 << 0) | (1 << 1) | (1 << 2)
        
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
