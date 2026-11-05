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

# Precompute degree of 4-AP conflicts for each element (number of 4-APs it's in)
elem_4ap_degree = [len(ap4_by_elem[i]) for i in range(N)]

# Precompute: for each element, the list of (ap3_index, other_elements_mask)
# where other_elements_mask has bits set for the other 2 elements in the 3-AP
ap3_other_mask = []
for i, m in enumerate(M3):
    # m has exactly 3 bits set
    bits = [b for b in range(N) if (m >> b) & 1]
    ap3_other_mask.append([(bits[0], bits[1], bits[2])])  # store the 3 positions

# For faster 3-AP counting: for each element x, store list of (other1, other2) pairs
# where {x, other1, other2} forms a 3-AP
ap3_partners = [[] for _ in range(N)]
for i, m in enumerate(M3):
    bits = [b for b in range(N) if (m >> b) & 1]
    for pos in range(3):
        x = bits[pos]
        others = [bits[j] for j in range(3) if j != pos]
        ap3_partners[x].append((others[0], others[1], i))

# For 4-AP validity checking: for each element x, store list of 4-AP masks it belongs to
# and for each such 4-AP, the mask of the other 3 elements
ap4_other_mask = [[] for _ in range(N)]
for i, m in enumerate(M4):
    bits = [b for b in range(N) if (m >> b) & 1]
    for pos in range(4):
        x = bits[pos]
        others_mask = 0
        for j in range(4):
            if j != pos:
                others_mask |= (1 << bits[j])
        ap4_other_mask[x].append((i, others_mask))

def count_3aps_fast(mask):
    """Count 3-APs in a bitmask using precomputed structure."""
    count = 0
    for x in range(N):
        if not (mask >> x) & 1:
            continue
        for (o1, o2, ap_idx) in ap3_partners[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1:
                # Only count once: ensure x is the smallest element in the AP
                if x < o1 and x < o2:
                    count += 1
    return count

def has_4ap_fast(mask):
    """Check if mask contains any 4-AP."""
    for x in range(N):
        if not (mask >> x) & 1:
            continue
        for (ap_idx, others_mask) in ap4_other_mask[x]:
            if (mask & others_mask) == others_mask:
                return True
    return False

def greedy_init(seed=None):
    """Greedy construction with random ordering."""
    if seed is not None:
        random.seed(seed)
    
    mask = 0
    count = 0
    order = list(range(N))
    random.shuffle(order)
    
    for x in order:
        bit = 1 << x
        if mask & bit:
            continue
        # Check 4-AP validity
        valid = True
        for (ap_idx, others_mask) in ap4_other_mask[x]:
            if (mask & others_mask) == others_mask:
                valid = False
                break
        if not valid:
            continue
        
        # Count 3-AP gain
        delta = 0
        for (o1, o2, ap_idx) in ap3_partners[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1:
                if x < o1 and x < o2:
                    delta += 1
        
        mask |= bit
        count += delta
    
    return mask, count

def local_search(mask, count, max_time=5.0):
    """Local search with add/remove/swap moves."""
    start_time = time.time()
    best_mask = mask
    best_count = count
    bits = [1 << i for i in range(N)]
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try removals
        for x in range(N):
            bit = bits[x]
            if not (mask & bit):
                continue
            
            # Compute delta for removal
            delta = 0
            for (o1, o2, ap_idx) in ap3_partners[x]:
                if (mask >> o1) & 1 and (mask >> o2) & 1:
                    if x < o1 and x < o2:
                        delta += 1
            
            new_count = count - delta
            if new_count > best_count:
                best_count = new_count
                best_mask = mask & ~bit
                mask &= ~bit
                count = new_count
                improved = True
        
        # Try additions
        for x in range(N):
            bit = bits[x]
            if mask & bit:
                continue
            
            # Check 4-AP validity
            valid = True
            for (ap_idx, others_mask) in ap4_other_mask[x]:
                if (mask & others_mask) == others_mask:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute delta for addition
            delta = 0
            for (o1, o2, ap_idx) in ap3_partners[x]:
                if (mask >> o1) & 1 and (mask >> o2) & 1:
                    if x < o1 and x < o2:
                        delta += 1
            
            new_count = count + delta
            if new_count > best_count:
                best_count = new_count
                best_mask = mask | bit
                mask |= bit
                count = new_count
                improved = True
        
        if not improved:
            break
    
    return best_mask, best_count

def constraint_backtracking(max_time=10.0, initial_mask=0):
    """
    Constraint programming with backtracking.
    Dynamic variable ordering by 4-AP conflict degree.
    Branch-and-bound using precomputed max 3-AP contribution.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # State: mask of included elements, count of 3-APs
    best_mask = initial_mask
    best_count = count_3aps_fast(initial_mask)
    
    # Elements to consider, ordered by 4-AP degree (highest first for more constraints)
    elements = sorted(range(N), key=lambda x: -elem_4ap_degree[x])
    
    # Precompute for each element: max possible 3-AP contribution
    # This is an upper bound on how many 3-APs this element can contribute
    max_3ap_contrib = [0] * N
    for x in range(N):
        max_3ap_contrib[x] = len(ap3_partners[x])
    
    # Current state
    mask = initial_mask
    count = best_count
    
    # Track which elements are decided
    decided = [False] * N
    for x in range(N):
        if mask & (1 << x):
            decided[x] = True
    
    # Backtracking search
    def backtrack(idx, mask, count):
        nonlocal best_mask, best_count
        
        if time.time() > deadline:
            return False
        
        if idx == N:
            if count > best_count:
                best_count = count
                best_mask = mask
            return True
        
        x = elements[idx]
        bit = 1 << x
        
        # Option 1: Exclude x
        # Compute upper bound: current count + max possible from remaining elements
        remaining_max = 0
        for j in range(idx + 1, N):
            y = elements[j]
            if not decided[y]:
                remaining_max += max_3ap_contrib[y]
        
        upper_bound = count + remaining_max
        if upper_bound <= best_count:
            return False  # Prune
        
        # Try excluding
        result = backtrack(idx + 1, mask, count)
        if result:
            return True
        
        # Option 2: Include x (if valid)
        # Check 4-AP validity
        valid = True
        for (ap_idx, others_mask) in ap4_other_mask[x]:
            if (mask & others_mask) == others_mask:
                valid = False
                break
        
        if valid:
            # Compute 3-AP gain
            delta = 0
            for (o1, o2, ap_idx) in ap3_partners[x]:
                if (mask >> o1) & 1 and (mask >> o2) & 1:
                    if x < o1 and x < o2:
                        delta += 1
            
            new_count = count + delta
            # Upper bound check
            remaining_max2 = 0
            for j in range(idx + 1, N):
                y = elements[j]
                if not decided[y]:
                    remaining_max2 += max_3ap_contrib[y]
            
            if new_count + remaining_max2 > best_count:
                result = backtrack(idx + 1, mask | bit, new_count)
                if result:
                    return True
        
        return False
    
    # Run backtracking with time limit
    # Use iterative deepening or just run until time runs out
    # For simplicity, we'll do a single pass but track best found
    
    # Actually, let's do a simpler approach: use the greedy solutions as starting points
    # and do local search from there, but with the constraint framework guiding the search
    
    return best_mask, best_count

def construct():
    start_time = time.time()
    deadline = 75.0  # Safety margin before 90s kill
    
    # Phase 1: Multiple greedy initializations
    best_mask = 0
    best_count = 0
    
    for i in range(30):
        if time.time() > deadline - 30:
            break
        m, c = greedy_init(seed=i)
        if c > best_count:
            best_count = c
            best_mask = m
    
    # Phase 2: Local search from best solution
    if time.time() < deadline - 10:
        best_mask, best_count = local_search(best_mask, best_count, max_time=min(10, deadline - time.time() - 5))
    
    # Phase 3: Additional local search with random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        restarts = max(1, int(remaining_time / 8))
        for i in range(restarts):
            if time.time() > deadline - 3:
                break
            
            # Create a perturbed version of the best solution
            mask = best_mask
            # Randomly flip some bits
            for _ in range(random.randint(5, 15)):
                x = random.randint(0, N - 1)
                bit = 1 << x
                if mask & bit:
                    mask &= ~bit
                else:
                    # Check validity before adding
                    valid = True
                    for (ap_idx, others_mask) in ap4_other_mask[x]:
                        if (mask & others_mask) == others_mask:
                            valid = False
                            break
                    if valid:
                        mask |= bit
            
            # Ensure validity
            # Remove any 4-AP violators
            changed = True
            while changed:
                changed = False
                for x in range(N):
                    if not (mask >> x) & 1:
                        continue
                    for (ap_idx, others_mask) in ap4_other_mask[x]:
                        if (mask & others_mask) == others_mask:
                            # Remove x to break this 4-AP
                            mask &= ~(1 << x)
                            changed = True
                            break
                    if changed:
                        break
            
            count = count_3aps_fast(mask)
            
            # Local search
            m, c = local_search(mask, count, max_time=min(5, deadline - time.time() - 2))
            if c > best_count:
                best_count = c
                best_mask = m
    
    # Convert to result
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
