"""Two Sidon sets with disjoint difference sets – Fast incremental search with local improvement."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 90s limit (evaluator kills at 90s, but let's be safe)


class SidonState:
    """Incremental state for checking Sidon property and difference disjointness."""
    
    def __init__(self, elements):
        self.elements = list(elements)
        self.sums = set()
        self.diffs = set()
        self._rebuild()
    
    def _rebuild(self):
        self.sums = set()
        self.diffs = set()
        for i, a in enumerate(self.elements):
            for b in self.elements[i:]:
                self.sums.add(a + b)
            for b in self.elements:
                d = b - a
                if d != 0:
                    self.diffs.add(d)
    
    def can_add(self, x, other_diffs):
        """Check if x can be added to this set. Uses set membership for O(1) lookups."""
        # Check sum collisions: for each existing a, a+x must not be in sums
        # Also x+x must not be in sums
        for a in self.elements:
            if (a + x) in self.sums:
                return False
        if (x + x) in self.sums:
            return False
        # Check difference collisions with other set
        for a in self.elements:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def add(self, x):
        """Add x to the set, updating sums and diffs."""
        self.elements.append(x)
        for a in self.elements[:-1]:  # all previous elements
            self.sums.add(a + x)
            d1 = x - a
            d2 = a - x
            self.diffs.add(d1)
            self.diffs.add(d2)
        self.sums.add(x + x)
    
    def remove(self, x):
        """Remove x from the set, updating sums and diffs."""
        self.elements.remove(x)
        # Rebuild is simpler and safer for correctness
        self._rebuild()
    
    def can_swap(self, x_out, x_in, other_diffs):
        """Check if we can swap x_out for x_in in this set."""
        if x_out not in self.elements:
            return False
        # Temporarily remove x_out
        temp_elements = [e for e in self.elements if e != x_out]
        temp_sums = set()
        temp_diffs = set()
        for i, a in enumerate(temp_elements):
            for b in temp_elements[i:]:
                temp_sums.add(a + b)
            for b in temp_elements:
                d = b - a
                if d != 0:
                    temp_diffs.add(d)
        # Now check if x_in can be added to temp set
        for a in temp_elements:
            if (a + x_in) in temp_sums:
                return False
        if (x_in + x_in) in temp_sums:
            return False
        for a in temp_elements:
            d1 = x_in - a
            d2 = a - x_in
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def do_swap(self, x_out, x_in):
        """Perform a swap: remove x_out, add x_in."""
        self.remove(x_out)
        self.add(x_in)


def greedy_build(order_a, order_b, deadline):
    """Build two Sidon sets greedily following given element orders."""
    state_a = SidonState([])
    state_b = SidonState([])
    
    for x in order_a:
        if time.time() > deadline:
            break
        if len(state_a.elements) >= MAX_SIZE:
            break
        if state_a.can_add(x, state_b.diffs):
            state_a.add(x)
    
    for x in order_b:
        if time.time() > deadline:
            break
        if len(state_b.elements) >= MAX_SIZE:
            break
        if state_b.can_add(x, state_a.diffs):
            state_b.add(x)
    
    return state_a.elements, state_b.elements


def greedy_build_interleaved(order_a, order_b, deadline):
    """Build two Sidon sets greedily, interleaving additions to A and B."""
    state_a = SidonState([])
    state_b = SidonState([])
    
    ia, ib = 0, 0
    while (ia < len(order_a) or ib < len(order_b)) and time.time() < deadline:
        # Try to add to A
        if ia < len(order_a) and len(state_a.elements) < MAX_SIZE:
            x = order_a[ia]
            ia += 1
            if state_a.can_add(x, state_b.diffs):
                state_a.add(x)
        # Try to add to B
        if ib < len(order_b) and len(state_b.elements) < MAX_SIZE:
            x = order_b[ib]
            ib += 1
            if state_b.can_add(x, state_a.diffs):
                state_b.add(x)
    
    return state_a.elements, state_b.elements


def local_improve(sets_a, sets_b, deadline):
    """Try to improve by adding or swapping elements."""
    a = list(sets_a)
    b = list(sets_b)
    state_a = SidonState(a)
    state_b = SidonState(b)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add new elements to A
        if len(a) < MAX_SIZE:
            for x in range(1, N + 1):
                if x in a:
                    continue
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    a.append(x)
                    improved = True
                    break
        
        # Try to add new elements to B
        if not improved and len(b) < MAX_SIZE:
            for x in range(1, N + 1):
                if x in b:
                    continue
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    b.append(x)
                    improved = True
                    break
        
        # Try swaps in A: remove one element, add another
        if not improved and len(a) > 0 and len(a) < MAX_SIZE:
            for x_out in list(a):
                if time.time() > deadline:
                    break
                for x_in in range(1, N + 1):
                    if x_in in a or x_in == x_out:
                        continue
                    if state_a.can_swap(x_out, x_in, state_b.diffs):
                        state_a.do_swap(x_out, x_in)
                        a.remove(x_out)
                        a.append(x_in)
                        improved = True
                        break
                if improved:
                    break
        
        # Try swaps in B: remove one element, add another
        if not improved and len(b) > 0 and len(b) < MAX_SIZE:
            for x_out in list(b):
                if time.time() > deadline:
                    break
                for x_in in range(1, N + 1):
                    if x_in in b or x_in == x_out:
                        continue
                    if state_b.can_swap(x_out, x_in, state_a.diffs):
                        state_b.do_swap(x_out, x_in)
                        b.remove(x_out)
                        b.append(x_in)
                        improved = True
                        break
                if improved:
                    break
    
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_a, best_b = [], []
    best_size = 0
    
    # Multiple restarts with different random orders
    random.seed(42)
    
    # Try both sequential and interleaved greedy strategies
    use_interleaved = False
    
    while time.time() < deadline - 2.0:  # Leave 2s margin
        # Create random permutations
        order_a = list(range(1, N + 1))
        random.shuffle(order_a)
        order_b = list(range(1, N + 1))
        random.shuffle(order_b)
        
        if use_interleaved:
            a, b = greedy_build_interleaved(order_a, order_b, deadline - 1.0)
        else:
            a, b = greedy_build(order_a, order_b, deadline - 1.0)
        
        current_size = len(a) + len(b)
        
        if current_size > best_size:
            best_size = current_size
            best_a, best_b = a, b
        
        # Try local improvement if we have time
        if time.time() < deadline - 3.0 and current_size >= best_size:
            a_imp, b_imp = local_improve(a, b, deadline - 1.0)
            imp_size = len(a_imp) + len(b_imp)
            if imp_size > best_size:
                best_size = imp_size
                best_a, best_b = a_imp, b_imp
        
        # Alternate between sequential and interleaved
        use_interleaved = not use_interleaved
    
    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]
