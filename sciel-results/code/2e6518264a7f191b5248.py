import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n_points = 16
    full_mask = (1 << n_points) - 1
    
    # 1. Generate all triples
    all_triples = []
    triple_masks = []
    for a in range(n_points):
        for b in range(a + 1, n_points):
            for c in range(b + 1, n_points):
                all_triples.append((a, b, c))
                mask = (1 << a) | (1 << b) | (1 << c)
                triple_masks.append(mask)
                
    n_triples = len(all_triples)
    
    # 2. Precompute compatibility masks
    # compat[i] is a bitmask of triples j such that triple i and j share at most 1 point.
    # We use Python integers as bitsets.
    compat = [0] * n_triples
    for i in range(n_triples):
        ti = triple_masks[i]
        for j in range(i + 1, n_triples):
            tj = triple_masks[j]
            if (ti & tj).bit_count() <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    # 3. Precompute supersets for each triple for fast count updates
    # supersets[i] is a list of masks that are supersets of triple i
    # Actually, iterating all 2^16 masks and checking if (m & ti) == ti is 65536 checks.
    # Better: generate supersets explicitly?
    # Number of supersets is 2^(16-3) = 8192.
    # We can just iterate all masks in the update step? 8192 is small.
    
    # State:
    # in_system: list of boolean or set of indices
    # count: array of length 2^16, count[m] = number of triples in system contained in m
    # alpha: current independence number
    
    def get_supersets(ti):
        """Generate all supersets of triple mask ti."""
        # ti has 3 bits set.
        # remaining bits = full_mask ^ ti
        # iterate submasks of remaining
        remaining = full_mask ^ ti
        sub = remaining
        while True:
            yield ti | sub
            if sub == 0:
                break
            sub = (sub - 1) & remaining
            
    # We need to be able to update count quickly.
    # Precomputing the list of supersets for each triple is memory heavy (560 * 8192 ints ~ 4.5M ints, ~36MB).
    # It's acceptable.
    superset_lists = []
    for i in range(n_triples):
        superset_lists.append(list(get_supersets(triple_masks[i])))
        
    def calculate_alpha(count):
        """Find max popcount of m where count[m] == 0."""
        max_alpha = 0
        for m in range(1 << n_points):
            if count[m] == 0:
                pc = m.bit_count()
                if pc > max_alpha:
                    max_alpha = pc
        return max_alpha

    def add_triple(idx, in_system, count):
        in_system[idx] = True
        for m in superset_lists[idx]:
            count[m] += 1

    def remove_triple(idx, in_system, count):
        in_system[idx] = False
        for m in superset_lists[idx]:
            count[m] -= 1

    # Greedy Initialization
    def greedy_init(rng):
        in_system = [False] * n_triples
        count = [0] * (1 << n_points)
        
        # Random order
        order = list(range(n_triples))
        rng.shuffle(order)
        
        for idx in order:
            # Check if compatible with all in_system
            # We can maintain a bitmask of available triples
            # But checking against in_system is O(N) per triple.
            # Better: maintain 'available' bitmask.
            # Initially all available.
            # When adding i, available &= compat[i].
            
            pass # We'll do a simpler greedy in the main loop for clarity, or just use a standard one.
            
        return in_system, count

    # Let's use a bitmask for available triples to speed up greedy selection.
    def greedy_construct(rng):
        in_system = [False] * n_triples
        count = [0] * (1 << n_points)
        available = (1 << n_triples) - 1  # All triples available initially
        
        while available:
            # Pick a random available triple
            # Get list of set bits
            bits = available.bit_count()
            if bits == 0:
                break
            k = rng.randrange(bits)
            # Find k-th set bit
            tmp = available
            for _ in range(k):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            idx = bit.bit_length() - 1
            
            add_triple(idx, in_system, count)
            available &= compat[idx]
            available &= ~(1 << idx) # Remove self
            
        return in_system, count

    best_alpha = 16
    best_in_system = None
    
    # Main Search Loop
    rng = random.Random(42)
    
    # Try multiple restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        in_system, count = greedy_construct(rng)
        current_alpha = calculate_alpha(count)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_in_system = in_system.copy()
            if best_alpha <= 7: # Very good, stop?
                break
        
        # Local Search / Simulated Annealing
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        # We will try swaps: remove i, add j
        # i must be in_system, j must be not in_system
        # j must be compatible with all in_system \ {i}
        
        # Precompute list of indices in system and not in system
        # This changes dynamically, so we recompute or maintain.
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Get current lists
            in_idx = [i for i in range(n_triples) if in_system[i]]
            if not in_idx:
                break
            
            # Pick a random triple to remove
            i = rng.choice(in_idx)
            
            # Remove it
            remove_triple(i, in_system, count)
            
            # Find candidates to add
            # A triple j is compatible if it is compatible with all k in in_system (excluding i)
            # We can check compatibility by checking if j is in the intersection of compat[k] for all k in in_system.
            # Since in_system size is ~100, intersection is expensive.
            # Alternative: Just check compatibility with each k in in_system.
            
            # But we can be smarter.
            # The set of valid j is: { j | j not in in_system AND for all k in in_system, (triple_masks[j] & triple_masks[k]).bit_count() <= 1 }
            
            # Let's just iterate all j not in in_system? That's 560 checks.
            # For each j, check against ~100 triples in system. 560 * 100 = 56k ops.
            # This is too slow for 500 steps * 1000 restarts.
            
            # Optimization:
            # Maintain a bitmask `compatible_with_all`?
            # When we remove i, we gain compatibility.
            # Let's just do a random search for a valid j.
            
            found_j = -1
            # Try a few random candidates
            for _ in range(20):
                j = rng.randrange(n_triples)
                if in_system[j]:
                    continue
                # Check compatibility
                ok = True
                for k in in_idx:
                    if k == i:
                        continue
                    if (triple_masks[j] & triple_masks[k]).bit_count() > 1:
                        ok = False
                        break
                if ok:
                    found_j = j
                    break
            
            if found_j == -1:
                # Restore i
                add_triple(i, in_system, count)
                continue
            
            # Add j
            add_triple(found_j, in_system, count)
            
            new_alpha = calculate_alpha(count)
            
            # Acceptance
            delta = new_alpha - current_alpha
            if delta <= 0:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_in_system = in_system.copy()
                    if best_alpha <= 7:
                        break
            else:
                # Simulated Annealing
                if rng.random() < math.exp(-delta / T):
                    current_alpha = new_alpha
                else:
                    # Reject: Restore state (remove j, add i)
                    remove_triple(found_j, in_system, count)
                    add_triple(i, in_system, count)
            
            T *= cooling
            if T < T_min:
                T = T_min

    # Reconstruct result from best_in_system
    result = []
    if best_in_system:
        for i in range(n_triples):
            if best_in_system[i]:
                result.append(list(all_triples[i]))
    else:
        # Fallback if something went wrong
        return []
        
    return result
