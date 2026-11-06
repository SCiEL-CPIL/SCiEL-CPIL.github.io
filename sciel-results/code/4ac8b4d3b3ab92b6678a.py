"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    # Bitset representation
    # chosen_bits: bit i (0-indexed) set if (i+1) is in chosen
    # We use 1-indexed: element x has bit (x-1) set
    chosen_bits = 0
    chosen = set()
    
    # sum_counts[s] = number of pairs (a,b) with a<=b, a+b=s, a,b in chosen
    sum_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    # blocked_bits: bit s set if sum_counts[s] >= 2
    blocked_bits = 0
    
    def can_add(x):
        """Check if adding x keeps all sums <= 2. O(1) with bitsets."""
        # Sums x+y for y in chosen: positions where (chosen_bits << (x-1)) has bits
        # Actually, if y is in chosen, bit (y-1) is set in chosen_bits
        # x+y = s means bit (s-1) in the shifted bitset
        # We want to check if any of these s have blocked_bits set
        # shifted = chosen_bits << (x-1) gives bits at positions (y-1)+(x-1) = x+y-2
        # So bit position p in shifted corresponds to sum = p+2
        # We need to check if bit (s-1) is set in blocked_bits for s = x+y
        # s-1 = x+y-1, and in shifted bitset, bit position is x+y-2
        # So we need blocked_bits shifted: bit (x+y-1) in blocked_bits
        # Equivalently: (blocked_bits >> (x-1)) & chosen_bits != 0
        # Because: bit y-1 in chosen_bits, we need bit x+y-1 in blocked_bits
        # blocked_bits >> (x-1) has bit (x+y-1-(x-1)) = y at position y-1... 
        # Let me redo: blocked_bits has bit s-1 set if sum s is blocked
        # We want to know if for any y in chosen, sum x+y is blocked
        # sum x+y is blocked iff bit (x+y-1) is set in blocked_bits
        # So we need: exists y in chosen such that bit (x+y-1) in blocked_bits
        # Shift blocked_bits right by (x-1): bit position p becomes p-(x-1)
        # Bit (x+y-1) becomes bit (x+y-1-(x-1)) = bit y
        # But chosen_bits has bit (y-1) set for element y
        # So we need bit y in (blocked_bits >> (x-1)) to correspond to bit (y-1) in chosen_bits
        # That's off by one. Let me use 0-indexed elements.
        
        # Let me redefine: element x (1-indexed) has bit index (x-1) in chosen_bits
        # sum s has bit index (s-1) in blocked_bits (or we can use s directly)
        # Let's use: bit index = value. So element x has bit x, sum s has bit s.
        # This wastes bit 0 but that's fine.
        
        # Redo with bit index = value:
        # chosen_bits: bit x set if x in chosen
        # blocked_bits: bit s set if sum_counts[s] >= 2
        # For candidate x: sums are x+y for y in chosen, and 2x
        # x+y: bit (x+y) in the result of (chosen_bits << x)
        # Check: (chosen_bits << x) & blocked_bits != 0
        # Also check: bit 2x in blocked_bits
        
        if (chosen_bits << x) & blocked_bits:
            return False
        if (blocked_bits >> (2*x)) & 1:
            return False
        return True
    
    def add_element(x):
        """Add x to chosen and update counts."""
        nonlocal chosen_bits, blocked_bits
        # Update sum_counts and blocked_bits
        for y in chosen:
            s = x + y
            sum_counts[s] += 1
            if sum_counts[s] == 2:
                blocked_bits |= (1 << s)
        s2 = 2 * x
        sum_counts[s2] += 1
        if sum_counts[s2] == 2:
            blocked_bits |= (1 << s2)
        chosen_bits |= (1 << x)
        chosen.add(x)
    
    def remove_element(x):
        """Remove x from chosen and update counts."""
        nonlocal chosen_bits, blocked_bits
        for y in chosen:
            if y == x:
                continue
            s = x + y
            sum_counts[s] -= 1
            if sum_counts[s] == 1:
                blocked_bits &= ~(1 << s)
        s2 = 2 * x
        sum_counts[s2] -= 1
        if sum_counts[s2] == 1:
            blocked_bits &= ~(1 << s2)
        chosen_bits &= ~(1 << x)
        chosen.discard(x)
    
    # Phase 1: Greedy construction
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if can_add(x):
            add_element(x)
    
    best_size = len(chosen)
    best_chosen = list(chosen)
    
    # Phase 2: Local search / simulated annealing
    # Try to swap elements to increase size
    rng = random.Random(42)
    
    def try_add_random():
        """Try to add a random element not in chosen."""
        candidates = [x for x in range(1, N + 1) if x not in chosen]
        if not candidates:
            return False
        x = rng.choice(candidates)
        if can_add(x):
            add_element(x)
            return True
        return False
    
    def try_swap():
        """Try to swap an element in chosen with one not in chosen."""
        if not chosen:
            return False
        x_out = rng.choice(list(chosen))
        remove_element(x_out)
        candidates = [x for x in range(1, N + 1) if x not in chosen]
        if not candidates:
            # Put it back
            add_element(x_out)
            return False
        # Try a few random candidates
        for _ in range(5):
            x_in = rng.choice(candidates)
            if can_add(x_in):
                add_element(x_in)
                return True
            candidates.remove(x_in)
        # Put it back
        add_element(x_out)
        return False
    
    # Simulated annealing style local search
    temp = 1.0
    cooling = 0.99
    min_temp = 0.01
    no_improve = 0
    
    while time.time() < deadline - 5 and temp > min_temp:
        if no_improve > 100:
            break
        
        if try_add_random():
            best_size = len(chosen)
            best_chosen = list(chosen)
            no_improve = 0
        elif try_swap():
            best_size = len(chosen)
            best_chosen = list(chosen)
            no_improve = 0
        else:
            no_improve += 1
        
        temp *= cooling
    
    return best_chosen
# EVOLVE-BLOCK-END
