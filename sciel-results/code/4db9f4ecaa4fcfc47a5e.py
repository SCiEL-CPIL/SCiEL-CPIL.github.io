"""Sum-free subset maximizer using multi-start greedy + local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def build_conflicts(elements):
        """Build set of conflicting pairs: for each c, pairs (a,b) where a+b=c and a,b,c in elements."""
        elem_set = set(elements)
        conflicts = {}  # c -> list of (a, b) pairs
        for c in elements:
            pairs = []
            for a in elements:
                b = c - a
                if b in elem_set:
                    pairs.append((a, b))
            if pairs:
                conflicts[c] = pairs
        return conflicts

    def is_sum_free(subset):
        """Check if subset is sum-free."""
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def greedy_initial(elements, order_seed, conflicts):
        """Greedy: add elements in given order if no conflict."""
        elem_set = set(elements)
        selected = set()
        # For each element, check if adding it creates a triple
        # A triple (a,b,c) with a+b=c is bad if all three are in selected
        # When adding c: check if there exist a,b in selected with a+b=c
        # When adding a: check if there exist b,c in selected with a+b=c or c-a=b
        for x in elements:
            # Check: is there a, b in selected such that a + b = x?
            conflict = False
            for a in selected:
                if (x - a) in selected:
                    conflict = True
                    break
            if not conflict:
                # Also check: x + b = c for b,c in selected (i.e., x = c - b)
                # and a + x = c for a,c in selected (i.e., c - x in selected)
                for b in selected:
                    if (x + b) in selected:
                        conflict = True
                        break
                    if (x - b) in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(x)
        return selected

    def local_search(selected, elements, max_iters=200):
        """Local search: try additions and swaps."""
        elem_set = set(elements)
        sel = set(selected)
        best_size = len(sel)
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding each unselected element
            for x in elements:
                if x in sel:
                    continue
                # Check if adding x creates a conflict
                ok = True
                for a in sel:
                    if (x - a) in sel:
                        ok = False
                        break
                    if (x + a) in sel:
                        ok = False
                        break
                if ok:
                    sel.add(x)
                    improved = True
                    if len(sel) > best_size:
                        best_size = len(sel)
            
            # Try 1-for-1 swaps: remove one, add one
            for x in elements:
                if x in sel:
                    continue
                for y in list(sel):
                    # Remove y, add x
                    sel.remove(y)
                    ok = True
                    for a in sel:
                        if (x - a) in sel:
                            ok = False
                            break
                        if (x + a) in sel:
                            ok = False
                            break
                    if ok:
                        sel.add(x)
                        improved = True
                        if len(sel) > best_size:
                            best_size = len(sel)
                    else:
                        sel.add(y)  # restore
                        break
            
            if not improved:
                break
        
        return sel

    def solve_one(elements):
        """Solve one instance with multiple greedy starts + local search."""
        n = len(elements)
        best = set()
        best_size = 0
        
        # Multiple greedy initializations with different orderings
        rng = random.Random(42)
        num_starts = min(5, n)
        
        for start_i in range(num_starts):
            if time.time() > DEADLINE - 5:
                break
            
            # Different orderings
            ordered = list(elements)
            if start_i == 0:
                pass  # sorted
            elif start_i == 1:
                rng.shuffle(ordered)
            elif start_i == 2:
                ordered.sort(key=lambda x: abs(x), reverse=True)
            elif start_i == 3:
                ordered.sort(key=lambda x: x)
            else:
                rng.shuffle(ordered)
            
            selected = greedy_initial(ordered, start_i, None)
            
            # Local search
            selected = local_search(selected, elements, max_iters=50)
            
            if len(selected) > best_size:
                best_size = len(selected)
                best = set(selected)
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE - 10:
            # Fallback: parity
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2])
            continue
        
        elements = instance(size, seed)
        selected = solve_one(elements)
        
        # Verify sum-free
        if not is_sum_free(selected):
            # Fallback to parity
            selected = set(x for x in elements if x % 2)
        
        results.append(sorted(selected))
    
    return results
