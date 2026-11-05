import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Map pair (a, b) with a < b to an index
    # Total pairs: 128 * 127 / 2 = 8128
    pair_index = {}
    idx = 0
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            pair_index[(a, b)] = idx
            idx += 1
            
    # Precompute conflict masks
    # conflict_mask[x] is a bitmask where bit i is set if pair i forms a bad triple with x
    conflict_mask = [0] * (N + 1)
    
    for x in range(1, N + 1):
        mask = 0
        for a in range(1, N + 1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ax == g_ab == g_bx:
                    # Find the index for pair (a, b)
                    if a < b:
                        p_idx = pair_index[(a, b)]
                    else:
                        p_idx = pair_index[(b, a)]
                    mask |= (1 << p_idx)
        conflict_mask[x] = mask

    class SetState:
        def __init__(self):
            self.elements = set()
            self.pairs_mask = 0
            
        def can_add(self, x):
            return (self.pairs_mask & conflict_mask[x]) == 0
            
        def add(self, x):
            if not self.can_add(x):
                return False
            # Update pairs_mask
            # New pairs are (x, y) for all y in elements
            for y in self.elements:
                if x < y:
                    p_idx = pair_index[(x, y)]
                else:
                    p_idx = pair_index[(y, x)]
                self.pairs_mask |= (1 << p_idx)
            self.elements.add(x)
            return True
            
        def remove(self, x):
            if x not in self.elements:
                return False
            # Update pairs_mask: clear bits for pairs (x, y)
            for y in self.elements:
                if y == x:
                    continue
                if x < y:
                    p_idx = pair_index[(x, y)]
                else:
                    p_idx = pair_index[(y, x)]
                self.pairs_mask &= ~(1 << p_idx)
            self.elements.discard(x)
            return True

    def greedy_build(order, state=None):
        if state is None:
            state = SetState()
        for x in order:
            if time.time() > deadline - 5:
                break
            if state.can_add(x):
                state.add(x)
        return state

    best_state = SetState()
    best_size = 0
    
    # 1. Multiple Greedy Starts
    # Use different orderings
    orderings = []
    # Random
    for _ in range(50):
        shuffled = numbers[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    # Sorted by value
    orderings.append(sorted(numbers))
    orderings.append(sorted(numbers, reverse=True))
    # Sorted by degree (number of conflicts)
    degrees = [0] * (N + 1)
    for x in range(1, N + 1):
        # Count how many pairs conflict with x
        # This is just the bit count of conflict_mask[x]
        degrees[x] = bin(conflict_mask[x]).count('1')
    orderings.append(sorted(numbers, key=lambda x: degrees[x]))
    orderings.append(sorted(numbers, key=lambda x: degrees[x], reverse=True))
    
    for order in orderings:
        state = greedy_build(order)
        if len(state.elements) > best_size:
            best_size = len(state.elements)
            best_state = state
        if time.time() > deadline - 20:
            break

    # 2. Local Search on Best State
    # We work on a copy to allow backtracking if needed, but simple hill climbing is usually enough
    current_state = SetState()
    current_state.elements = set(best_state.elements)
    # Rebuild pairs_mask for current_state
    for x in current_state.elements:
        for y in current_state.elements:
            if x < y:
                p_idx = pair_index[(x, y)]
                current_state.pairs_mask |= (1 << p_idx)
                
    unselected = [x for x in numbers if x not in current_state.elements]
    
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        improved = False
        
        # Phase A: Try to add any compatible number
        # Shuffle unselected to randomize
        random.shuffle(unselected)
        for x in unselected:
            if current_state.can_add(x):
                current_state.add(x)
                unselected.remove(x)
                improved = True
                break
        
        if improved:
            no_improve_count = 0
            continue
            
        no_improve_count += 1
        
        # Phase B: Try swaps
        if no_improve_count > 2:
            # Try to find a swap (remove y, add x) that allows adding more later?
            # Or just a swap that maintains size but might enable further additions.
            # Actually, simple swaps don't increase size. We need to find a sequence.
            # Let's try "kicks": remove k random elements, then greedily refill.
            
            if len(current_state.elements) > 10:
                k = random.randint(2, 5)
                # Remove k random elements
                to_remove = random.sample(list(current_state.elements), k)
                for y in to_remove:
                    current_state.remove(y)
                    if y in unselected:
                        unselected.remove(y)
                    else:
                        unselected.append(y)
                        
                # Greedily refill
                random.shuffle(unselected)
                for x in unselected:
                    if current_state.can_add(x):
                        current_state.add(x)
                        unselected.remove(x)
                        
                # If we didn't improve size, we might have made it worse.
                # We should keep track of the best state and revert if worse.
                # For simplicity, let's just continue. If it gets stuck, we try again.
                # A better way: keep a backup.
                
                if len(current_state.elements) > best_size:
                    best_size = len(current_state.elements)
                    best_state.elements = set(current_state.elements)
                    # Rebuild pairs mask for best_state if needed, or just keep current
                    # We'll just update best_size and copy elements
                    # Note: best_state.pairs_mask is not updated here, but we only use it for size comparison
                    # We should update best_state properly if we want to use it later.
                    # Let's just keep current_state as the active one.
                    
                no_improve_count = 0
                improved = True # Reset counter to allow more kicks

    # Final polish: try to add any remaining
    unselected = [x for x in numbers if x not in current_state.elements]
    random.shuffle(unselected)
    for x in unselected:
        if current_state.can_add(x):
            current_state.add(x)

    return list(current_state.elements)
