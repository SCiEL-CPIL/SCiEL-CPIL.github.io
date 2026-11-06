import time
import random
import sys

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin for 60s limit

class SidonState:
    """
    Tracks elements, pair sums, and absolute nonzero differences using Python integers as bitmasks.
    Sum mask: bit s set if sum s is present. Range 0..512.
    Diff mask: bit d set if absolute difference d is present. Range 0..255.
    """
    def __init__(self):
        self.els = set()
        self.sum_mask = 0
        self.diff_mask = 0
        
    def can_add(self, x, other_diff_mask=0):
        if x in self.els:
            return False
        
        # Check sums:
        # 1. x+x collision
        if (self.sum_mask >> (2*x)) & 1:
            return False
        
        # 2. a+x collisions
        # We need to check if any (a+x) is in self.sum_mask.
        # Iterate is O(N), but N <= 64.
        # Optimization: Precompute element bitmask?
        # If we have elem_mask, we can do:
        # candidate_sums_mask = 0
        # for a in els: candidate_sums_mask |= (1 << (a+x))
        # if candidate_sums_mask & self.sum_mask: return False
        # This is still O(N) to build candidate_sums_mask.
        
        # Check diffs:
        # New diffs: |x-a| for a in els.
        # These must not be in self.diff_mask or other_diff_mask.
        
        for a in self.els:
            s = x + a
            if (self.sum_mask >> s) & 1:
                return False
            
            d = abs(x - a)
            if (self.diff_mask >> d) & 1:
                return False
            if (other_diff_mask >> d) & 1:
                return False
                
        return True

    def add(self, x):
        self.els.add(x)
        self.sum_mask |= (1 << (2*x))
        for a in self.els:
            if a != x:
                self.sum_mask |= (1 << (a + x))
                d = abs(x - a)
                self.diff_mask |= (1 << d)

    def remove(self, x):
        if x not in self.els:
            return
        self.els.discard(x)
        # Rebuild masks from scratch is O(N^2) but N is small.
        # Or clear bits. Since Sidon sets have unique sums and unique abs diffs?
        # Unique sums: Yes.
        # Unique abs diffs: Yes, for Sidon sets, all non-zero differences are distinct in absolute value.
        # Proof: If |a-b| = |c-d| = k, then either a-b=k and c-d=k => a+d = b+c.
        # If {a,d} != {b,c}, sum collision.
        # If a=c and d=b, then a-b = c-d.
        # If a=d and c=b, then a-b = -(c-d) => |a-b|=|c-d|.
        # Wait. If A={1,2,4}, diffs: 1, 3, 2. All distinct.
        # If A={1,3,4}, diffs: 2, 3, 1. All distinct.
        # Yes, abs diffs are unique in a Sidon set.
        # So we can just clear bits.
        
        self.sum_mask &= ~(1 << (2*x))
        for a in self.els:
            self.sum_mask &= ~(1 << (a + x))
            d = abs(x - a)
            self.diff_mask &= ~(1 << d)

    def size(self):
        return len(self.els)

    def copy(self):
        new_state = SidonState()
        new_state.els = set(self.els)
        new_state.sum_mask = self.sum_mask
        new_state.diff_mask = self.diff_mask
        return new_state

def greedy_build(order, other_state=None, max_size=MAX_SIZE, deadline=None):
    state = SidonState()
    other_diff_mask = other_state.diff_mask if other_state else 0
    for x in order:
        if deadline and time.time() > deadline:
            break
        if state.size() >= max_size:
            break
        if state.can_add(x, other_diff_mask):
            state.add(x)
    return state

def local_search_polish(A_state, B_state, deadline):
    """
    Try to improve by adding any remaining element.
    If stuck, try 1-for-1 swaps that might enable future adds? 
    For now, just try to add. If we can add, great.
    """
    improved = True
    all_candidates = list(range(1, N + 1))
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add to A
        if A_state.size() < MAX_SIZE:
            candidates = [x for x in all_candidates if x not in A_state.els]
            random.shuffle(candidates)
            for x in candidates:
                if A_state.can_add(x, B_state.diff_mask):
                    A_state.add(x)
                    improved = True
                    break
        
        # Try to add to B
        if B_state.size() < MAX_SIZE:
            candidates = [x for x in all_candidates if x not in B_state.els]
            random.shuffle(candidates)
            for x in candidates:
                if B_state.can_add(x, A_state.diff_mask):
                    B_state.add(x)
                    improved = True
                    break
                    
    return A_state, B_state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_A = set()
    best_B = set()
    best_total = 0
    
    # Use a fixed seed for reproducibility of the "best" structure, 
    # but randomize the ILS process.
    rng = random.Random(42)
    
    # Phase 1: Pure ILS (Iterated Local Search)
    # Greedy + Polish
    
    while time.time() < deadline - 2.0:
        # Generate random orders
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        
        # Alternate building to break symmetry
        if rng.random() > 0.5:
            A_state = greedy_build(order_a, other_state=None, deadline=deadline)
            B_state = greedy_build(order_b, other_state=A_state, deadline=deadline)
        else:
            B_state = greedy_build(order_b, other_state=None, deadline=deadline)
            A_state = greedy_build(order_a, other_state=B_state, deadline=deadline)
        
        # Polish
        A_state, B_state = local_search_polish(A_state, B_state, deadline)
        
        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)
            
            # If we reach a very high score, we might want to stop early?
            # Max possible is 128. 
            # Theoretical bound for 2 Sidon sets with disjoint diffs in [1,256]:
            # Each Sidon set size ~ sqrt(256) = 16? No, max Sidon set in [1,n] is ~ sqrt(n).
            # For n=256, max Sidon size is around 16-18?
            # Wait, Sidon set size bound: |A| <= sqrt(n) + O(1).
            # sqrt(256) = 16.
            # So max |A| is ~16. Max |A|+|B| is ~32.
            # If we hit 32, we are near optimal.
            if best_total >= 32:
                # Try a few more to see if 33 is possible? Unlikely.
                # But let's keep running to be safe, maybe 31 is the max?
                pass

    # Fallback
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}
        
    return [sorted(best_A), sorted(best_B)]

if __name__ == "__main__":
    A, B = construct()
    print(f"Sizes: {len(A)}, {len(B)}, Total: {len(A)+len(B)}")
