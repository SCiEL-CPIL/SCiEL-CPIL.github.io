import time
import random
import sys
import numpy as np

N = 256
MAX_SIZE = 64
DEADLINE = 50.0  # Safety margin for 60s limit

# Bitmask sizes:
# Sums: max sum is 256+256=512. We need bits 0..511.
# Diffs: max diff is 255, min is -255. Range 0..510 (offset by 255).
# We use uint64 bitmasks.
# Sum index: s
# Diff index: d + 255

class SidonState:
    """
    Tracks elements, pair sums, and nonzero differences using NumPy bitmasks.
    """
    def __init__(self):
        self.els = set()
        # sum_mask: bit i set if sum i is present
        self.sum_mask = 0
        # diff_mask: bit (d+255) set if diff d is present
        self.diff_mask = 0
        
    def can_add(self, x, other_diff_mask=0):
        if x in self.els:
            return False
        
        # Check sums:
        # New sums: x+x, and a+x for all a in els
        # We can construct a bitmask of new sums and check intersection with self.sum_mask
        # But iterating is still O(N). Let's try to be smart.
        # Actually, since N=256, O(N) is fine if constant is small.
        # However, we can precompute candidate sum mask? No, depends on x.
        
        # Faster check:
        # 1. x+x collision
        if (self.sum_mask >> (2*x)) & 1:
            return False
        
        # 2. a+x collisions
        # For each a in els, check if (a+x) in sum_mask
        # We can do this by shifting sum_mask?
        # If we have sum_mask, and we want to check if any (a+x) is in sum_mask.
        # This is equivalent to: (sum_mask >> x) & element_mask != 0 ?
        # Let element_mask be bitmask of elements.
        # If (sum_mask >> x) & element_mask != 0, then there exists a such that a+x is in sums.
        # Wait. sum_mask has bit s if s is a sum.
        # We want to know if there is a in els such that a+x is in sums.
        # This means there is a in els such that bit (a+x) in sum_mask is 1.
        # Let E be bitmask of elements.
        # (sum_mask >> x) has bit a set if bit (a+x) in sum_mask is set.
        # So if ((sum_mask >> x) & E) != 0, collision.
        
        elem_mask = 0
        for a in self.els:
            elem_mask |= (1 << a)
            
        if ((self.sum_mask >> x) & elem_mask) != 0:
            return False
            
        # Check diffs:
        # New diffs: x-a and a-x for all a in els.
        # We need to check if any of these are in self.diff_mask or other_diff_mask.
        # Diff d corresponds to bit d+255.
        # d = x - a => bit index x - a + 255
        # d = a - x => bit index a - x + 255
        
        # Check against self.diff_mask:
        # For each a in els:
        #   d1 = x - a. Check bit d1+255 in self.diff_mask.
        #   d2 = a - x. Check bit d2+255 in self.diff_mask.
        
        # Vectorized check for self.diff_mask:
        # We can precompute a bitmask for potential diffs?
        # Let's just iterate for diffs, it's usually the bottleneck for logic, but bitmask ops are fast.
        # Actually, we can do:
        # diff_mask_shifted_pos = self.diff_mask >> (x + 255) ? No.
        
        # Let's stick to simple iteration for diffs but use bitmask for sums which is the bigger set.
        # Or just iterate both. The main gain is in local search trying many candidates.
        
        for a in self.els:
            d1 = x - a
            d2 = a - x
            # Check self
            if (self.diff_mask >> (d1 + 255)) & 1:
                return False
            if (self.diff_mask >> (d2 + 255)) & 1:
                return False
            # Check other
            if other_diff_mask:
                if (other_diff_mask >> (d1 + 255)) & 1:
                    return False
                if (other_diff_mask >> (d2 + 255)) & 1:
                    return False
        return True

    def add(self, x):
        self.els.add(x)
        # Update sum_mask
        self.sum_mask |= (1 << (2*x))
        for a in self.els:
            if a != x:
                self.sum_mask |= (1 << (a + x))
                # Update diff_mask
                d1 = x - a
                d2 = a - x
                self.diff_mask |= (1 << (d1 + 255))
                self.diff_mask |= (1 << (d2 + 255))

    def remove(self, x):
        if x not in self.els:
            return
        # We must rebuild masks or carefully clear bits.
        # Clearing bits is tricky if multiple pairs produce same sum/diff.
        # But Sidon sets have unique sums! So sum_mask bits are unique per pair?
        # Yes, pair sums are unique. So we can just clear the bits corresponding to pairs involving x.
        # Diffs: Nonzero differences in a Sidon set are NOT necessarily unique.
        # Wait, Sidon set definition: all pair sums a+b (a<=b) are distinct.
        # This implies all differences a-b are distinct?
        # If a-b = c-d, then a+d = c+b. If {a,d} != {c,b}, this is a sum collision.
        # If a=d and c=b, then a-b = a-b.
        # If a=c and d=b, then a-b = a-b.
        # So yes, in a Sidon set, all nonzero differences are distinct (up to sign).
        # Actually, if a-b = c-d, then a+d = b+c.
        # If {a,d} = {b,c}, then a=b and d=c => a-b=0.
        # Or a=c and d=b => a-b = c-d.
        # So distinct differences are guaranteed for Sidon sets.
        # Thus, we can safely clear bits.
        
        self.els.discard(x)
        self.sum_mask &= ~(1 << (2*x))
        for a in self.els:
            self.sum_mask &= ~(1 << (a + x))
            d1 = x - a
            d2 = a - x
            self.diff_mask &= ~(1 << (d1 + 255))
            self.diff_mask &= ~(1 << (d2 + 255))

    def size(self):
        return len(self.els)

    def copy(self):
        new_state = SidonState()
        new_state.els = set(self.els)
        new_state.sum_mask = self.sum_mask
        new_state.diff_mask = self.diff_mask
        return new_state

def greedy_build(order, other_state=None, max_size=MAX_SIZE):
    state = SidonState()
    other_diff_mask = other_state.diff_mask if other_state else 0
    for x in order:
        if state.size() >= max_size:
            break
        if state.can_add(x, other_diff_mask):
            state.add(x)
    return state

def local_search(A_state, B_state, deadline, rng):
    """
    Local search with Add, Swap, and Move operations using bitmask checks.
    """
    improved = True
    # Precompute all candidates
    all_candidates = list(range(1, N + 1))
    
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Pure Additions
        # Add to A
        if A_state.size() < MAX_SIZE:
            candidates = [x for x in all_candidates if x not in A_state.els]
            rng.shuffle(candidates)
            for x in candidates:
                if A_state.can_add(x, B_state.diff_mask):
                    A_state.add(x)
                    improved = True
                    break
        
        # Add to B
        if B_state.size() < MAX_SIZE:
            candidates = [x for x in all_candidates if x not in B_state.els]
            rng.shuffle(candidates)
            for x in candidates:
                if B_state.can_add(x, A_state.diff_mask):
                    B_state.add(x)
                    improved = True
                    break

        # 2. Try Swaps (Remove one, Add one) if stuck or to optimize
        if not improved:
            # Swap in A
            if A_state.size() > 0 and A_state.size() < MAX_SIZE:
                a_list = list(A_state.els)
                rng.shuffle(a_list)
                for a_out in a_list:
                    A_state.remove(a_out)
                    # Try to add something new to A
                    candidates = [x for x in all_candidates if x not in A_state.els and x != a_out]
                    rng.shuffle(candidates)
                    added = False
                    for x_in in candidates:
                        if A_state.can_add(x_in, B_state.diff_mask):
                            A_state.add(x_in)
                            added = True
                            improved = True
                            break
                    if not added:
                        # Revert
                        A_state.add(a_out)
                    else:
                        break # Restart loop with new state
            
            # Swap in B
            if not improved and B_state.size() > 0 and B_state.size() < MAX_SIZE:
                b_list = list(B_state.els)
                rng.shuffle(b_list)
                for b_out in b_list:
                    B_state.remove(b_out)
                    candidates = [x for x in all_candidates if x not in B_state.els and x != b_out]
                    rng.shuffle(candidates)
                    added = False
                    for x_in in candidates:
                        if B_state.can_add(x_in, A_state.diff_mask):
                            B_state.add(x_in)
                            added = True
                            improved = True
                            break
                    if not added:
                        B_state.add(b_out)
                    else:
                        break

        # 3. Try Moves (Move element from A to B or B to A)
        if not improved:
            # Move from A to B
            if A_state.size() > 0 and B_state.size() < MAX_SIZE:
                a_list = list(A_state.els)
                rng.shuffle(a_list)
                for a in a_list:
                    A_state.remove(a)
                    if B_state.can_add(a, A_state.diff_mask):
                        B_state.add(a)
                        improved = True
                        break
                    else:
                        A_state.add(a) # Revert

            # Move from B to A
            if not improved and B_state.size() > 0 and A_state.size() < MAX_SIZE:
                b_list = list(B_state.els)
                rng.shuffle(b_list)
                for b in b_list:
                    B_state.remove(b)
                    if A_state.can_add(b, B_state.diff_mask):
                        A_state.add(b)
                        improved = True
                        break
                    else:
                        B_state.add(b) # Revert

    return A_state, B_state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_A = set()
    best_B = set()
    best_total = 0
    
    rng = random.Random(42)
    
    # Multiple restarts
    max_restarts = 200
    for trial in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random orders
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        
        # Alternate building order to break symmetry
        if trial % 2 == 0:
            A_state = greedy_build(order_a, other_state=None)
            B_state = greedy_build(order_b, other_state=A_state)
        else:
            B_state = greedy_build(order_b, other_state=None)
            A_state = greedy_build(order_a, other_state=B_state)
        
        # Local search
        A_state, B_state = local_search(A_state, B_state, deadline - 2.0, rng)
        
        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)
            
            # Early stop if we hit theoretical bound? 
            # Theoretical max is around 32. If we hit 30+, we are good.
            if best_total >= 30:
                # We can try a few more to see if we can push to 32, but 30 is already great.
                # Let's run a few more restarts to be sure.
                pass

    # Fallback if nothing found (should not happen)
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}
        
    # Ensure caps
    if len(best_A) > MAX_SIZE:
        best_A = set(list(best_A)[:MAX_SIZE])
    if len(best_B) > MAX_SIZE:
        best_B = set(list(best_B)[:MAX_SIZE])

    return [sorted(best_A), sorted(best_B)]

if __name__ == "__main__":
    A, B = construct()
    print(f"Sizes: {len(A)}, {len(B)}, Total: {len(A)+len(B)}")
